import copy
import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR 

from functools import partial
from tqdm.autonotebook import trange

import matplotlib.pyplot as plt
from matplotlib import cm


# Evaluation metrics
max_error = lambda u_exact, u_approx: torch.norm(
        u_exact - u_approx, p=float('inf'))
avg_error = lambda u_exact, u_approx: torch.mean(
        (u_exact - u_approx)**2)**.5


def minstrX(xdata, ydata):
    x_min = xdata[ydata.argmin()]
    m, e = f'{x_min:.2e}'.split('e')
    x_min_str = r'${}\times 10^{{{:d}}}$'.format(m, int(e))
    return x_min_str, x_min


class SepLossTrainer:
    def __init__(self, net, pde, optimizer, scheduler,
            refinement=None, k=None, pbar=False):
        """
        Parameters:
            k : int, None by default
                hyperparameter in the refinement procedure

            refinement: str ('RAR'/'GAR') or None
                'RAR' - residual-based adaptive refinement
                'GAR' - gradient-based adaptive refinement

            pbar : bool flag
                Switch on/off progress bar for epoch training
        """
        self.No = 0
        self.best_epoch = 0
        self.best_score = float('inf')
        self.best_weights = None
        self.history = []

        self.net = net
        self.pde = pde
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.meta = None
        self.refinement = (refinement, k)

        self._sbatch = None
        self._history = []
        self._eps_0, self._eps_r = 1., 0.
        self._iter = lambda n,s: trange(n, desc=s)
        if not pbar: self._iter = lambda n, s: range(n)

    def histories(self):
        """
        Returns the set of loss histories
        A separate loss history can be obtained by row slicing
        """
        return np.array(self._history).T

    @property
    def refinement(self):
        return self.__refinement

    @refinement.setter
    def refinement(self, x):
        try:
            r, k = x
        except ValueError:
            r, k = x, None
        except TypeError:
            r, k = x, None

        if k is not None:
            assert k >= 1, '[ $k -ge 1 ] must be true'

        if r is None:
            self.__refinement = 'No refinement'
            self.__closure = self._closure
            self.__k = None
        elif r == 'RAR':
            self.__refinement = r
            self.__closure = self._closureRAR
            if k is None: k = 2
            self.__k = k
        elif r == 'GAR':
            self.__refinement = r
            self.__closure = self._closureGAR
            if k is None: k = 4
            self.__k = k
        else:
            raise ValueError(
                    'Invalid refinement procedure.\n'
                    'If k is set, then you must specify'
                    "'RAR' or 'GAR' as refinement named argument")

    @refinement.getter
    def refinement(self):
        if getattr(self, '_SepLossTrainer__k', None):
            return self.__refinement, self.__k
        return self.__refinement

    def trainOneEpoch(self, w, num_batches=100, batch_size=128, lp=1):
        """
        w : torch.Tensor transfered on the device
            weights for the manual balancing of loss terms
        lp: int, log period
        """
        loss_logs = w.new_empty([num_batches, w.nelement()])
        for i in self._iter(num_batches, f'Epoch {self.No}'):
            self.__closure(w, batch_size, loss_logs, i)

        self.scheduler.step()
        if self.No % lp == 0:
            self._history.append(loss_logs.mean(0).cpu().numpy())
            self.history.append(self._history[-1].sum())
        self.No += 1

    def _closure(self, w, batch_size, loss_logs, i):
        self.optimizer.zero_grad()

        batch = self.pde.sampleBatch(batch_size)
        L = self.pde.computeLoss(batch, self.net)
        loss_logs[i] = L.data
        (w@L).backward()

        self.optimizer.step()

    def _closureRAR(self, w, batch_size, loss_logs, i):
        self.optimizer.zero_grad()
        if self._eps_r < self._eps_0:
            self._sbatch, self._eps_0 = self._makeSuppBatchR(batch_size)

        batch = self.pde.sampleBatch(batch_size)
        batch_2x = torch.cat([batch, self._sbatch])
        L = self.pde.computeLoss(batch_2x, self.net)
        loss_logs[i] = L.data
        (w@L).backward()

        self.optimizer.step()
        self._eps_r = self.pde.computeLoss(
                self._sbatch, self.net)[0].item()

    def _closureGAR(self, w, batch_size, loss_logs, i):
        self.optimizer.zero_grad()
        self._sbatch = self._makeSuppBatchG(batch_size)

        batch = self.pde.sampleBatch(batch_size)
        batch_4x = torch.cat([batch, self._sbatch])
        L = self.pde.computeLoss(batch_4x, self.net)
        loss_logs[i] = L.data
        (w@L).backward()

        self.optimizer.step()

    def _makeSuppBatchR(self, batch_size):
        """
        Residual-based Adaptive Refinement
        Section 2.8, https://arxiv.org/pdf/1907.04502.pdf
        ---
        Unfortunately, in the current implementation,
        it is only possible to make differential op-r refinement
        since for all loss parts there is a single batch generated
        """
        batch = self.pde.sampleBatch(self.__k*batch_size)
        R_do,*_ = self.pde.computeResiduals(batch, self.net)
        values, indices = R_do.detach().sort()
        support_batch = batch[indices[-batch_size:]]
        eps_0 = values[:batch_size].mean().item()
        return support_batch, eps_0

    def _makeSuppBatchG(self, batch_size):
        """
        Gradient-based Adaptive Refinement
        (my implementation, tell me if sth better exists)
        ---
        One calculates the gradients only in the domain
        interior, since the gradients on the borders are
        the part of the boundary conditions
        """
        batch = self.pde.sampleBatch(self.__k*batch_size)
        GNs = self.pde.computeGradNorms(batch, self.net)
        _, indices = GNs.detach().sort()
        support_batch = batch[indices[-batch_size:]]
        jitter = 5e-3*self.pde._len * batch.new(
                3, *support_batch.shape).normal_()

        support_batch = (support_batch + jitter).flatten(0, 1)
        support_batch = torch.max(support_batch, self.pde.lims[:, 0])
        support_batch = torch.min(support_batch, self.pde.lims[:, 1])
        return support_batch

    def linearWarmUp(
            self, w, batch_size, num_iters=1000,
            lr_lims=None, range_test=False, explosion_ratio=None):
        """
        Network warm up with linear increasing LR from some small value
        If `range_test` is `True`, then test on more relevant LR will be
        run instead

        With `explosion_ratio`, you can reduce the method's execution time
        by setting the argument's value close to one. It limits the maximum
        possible ration between the total loss at some iteration and
        at the first step
        """
        assert explosion_ratio > 1, 'No sense to set it smaller than 1'
        loss_logs = w.new_empty([num_iters, w.nelement()])
        opt_dict = copy.deepcopy(self.optimizer.state_dict())
        if range_test:
            net_weights = copy.deepcopy(self.net.state_dict())

        if lr_lims is not None:
            lr_0, lr_n = lr_lims
            assert lr_n > lr_0, 'Inappropriate range'
        else:
            lr_n = self.scheduler.base_lrs[0]
            assert lr_n != 0, 'Zero optimizer lr'
            lr_0 = lr_n / num_iters
        xdata = np.linspace(lr_0, lr_n, num_iters, dtype='f4')

        desc = 'Range test' if range_test else 'Warm-up'
        for i in trange(num_iters, desc=desc):
            self.optimizer.zero_grad()
            for par in self.optimizer.param_groups:
                par['lr'] = xdata[i]
            batch = self.pde.sampleBatch(batch_size)
            L = self.pde.computeLoss(batch, self.net)
            loss_logs[i] = L.data
            (w@L).backward()

            self.optimizer.step()
            if (explosion_ratio is not None
                    and explosion_ratio*(
                        w@loss_logs[0]).item() < (w@L).item()):
                print('Early stop due to the loss explosion')
                break
        self.optimizer.load_state_dict(opt_dict)
        if range_test:
            self.net.load_state_dict(net_weights)
            args = ('linear', explosion_ratio, w, i)
            self._displayRTresults(xdata, loss_logs, *args)

    def exponentialWarmUp(
            self, w, batch_size, num_iters=1000,
            lr_lims=None, range_test=False, explosion_ratio=None):
        """
        Network warm up with exponentailly increasing LR from some small value
        If `range_test` is `True`, then test on more relevant LR will be
        run instead

        With `explosion_ratio`, you can reduce the method's execution time
        by setting the argument's value close to one. It limits the maximum
        possible ration between the total loss at some iteration and
        at the first step
        """
        loss_logs = w.new_empty([num_iters, w.nelement()])
        opt_dict = copy.deepcopy(self.optimizer.state_dict())
        if range_test:
            net_weights = copy.deepcopy(self.net.state_dict())

        if lr_lims is not None:
            lr_0, lr_n = lr_lims
            assert lr_n > lr_0 and lr_0 > 0, 'Inappropriate range'
            gamma = (lr_n/lr_0)**(1./num_iters)
        else:
            lr_n = self.scheduler.base_lrs[0]
            assert lr_n != 0, 'Zero optimizer lr'
            lr_0 = lr_n*1e-5
            gamma = (1e5)**(1./num_iters)

        for par in self.optimizer.param_groups:
            par['lr'] = lr_0
        scheduler = StepLR(self.optimizer, 1, gamma)
        xdata = np.empty(num_iters, 'f4')

        desc = 'Range test' if range_test else 'Warm-up'
        for i in trange(num_iters, desc=desc):
            self.optimizer.zero_grad()
            xdata[i] = scheduler.get_last_lr()[0]

            batch = self.pde.sampleBatch(batch_size)
            L = self.pde.computeLoss(batch, self.net)
            loss_logs[i] = L.data
            (w@L).backward()

            self.optimizer.step()
            scheduler.step()
            if (explosion_ratio is not None
                    and explosion_ratio*(
                        w@loss_logs[0]).item() < (w@L).item()):
                print('Early stop due to the loss explosion')
                break
        self.optimizer.load_state_dict(opt_dict)
        if range_test:
            self.net.load_state_dict(net_weights)
            args = ('expo', explosion_ratio, w, i)
            self._displayRTresults(xdata, loss_logs, *args)

    def _displayRTresults(
            self, xdata, loss_logs, mode, explosion_ratio, w, i):
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(9, 4))
        ax2 = ax1.twinx()
        ax4 = ax3.twinx()

        xdata = xdata[:i+1]
        loss_logs = loss_logs.cpu().numpy()[:i+1].T
        mllog = loss_logs.mean(0)
        wllog = w.cpu().numpy() @ loss_logs
        x_min_str, x_min = minstrX(xdata, mllog)
        x_wmin_str, x_wmin = minstrX(xdata, wllog)

        viridis = cm.get_cmap('viridis', len(loss_logs))
        cmap = viridis(np.linspace(0, 1, len(loss_logs)))

        lbls = ['do', 'ic']
        repeats = len(loss_logs) - len(lbls)
        lbls += ['bc'] * repeats
        axes = [ax1, ax2] + [ax2] * repeats
        suffix = 'es' if repeats else ''

        for lbl, c, ax, loss in zip(lbls, cmap, axes, loss_logs):
            ax.plot(xdata, loss, c=c, alpha=.4, label=lbl)
        lines = ax1.lines + ax2.lines
        fig.suptitle('LR exponential range test')
        ax1.tick_params(axis='y', labelcolor=cmap[0])
        ax1.axvline(x_min, 0, 1, c='red', ls='--', alpha=.5)
        ax1.axvline(x_wmin, 0, 1, c='pink', ls='--', alpha=.5)
        ax2.set_ylabel(','.join(lbls[1:])+' loss'+suffix)
        ax1.set_ylabel('do loss')
        ax1.set_xlabel('iteration #')
        ax1.legend(lines, lbls, loc=9);
        if mode == 'expo':
            ax1.set_xscale('log')
            ax2.set_xscale('log')
            ax3.set_xscale('log')

        ax3.plot(xdata, mllog, alpha=.6, c='dodgerblue')
        ax4.plot(xdata, wllog, alpha=.6, c='darkorange')
        lines = ax3.lines+ax4.lines
        lbls = ['arithmetic mean', 'weighted']
        ax3.legend(lines, lbls, loc=9)

        ax3.set_ylabel('mean loss')
        ax4.set_ylabel("loss terms' weighted sum")
        ax3.axvline(x_min, 0, 1, c='red', ls='--', alpha=.5)
        ax3.axvline(x_wmin, 0, 1, c='pink', ls='--', alpha=.5)
        ax3.text(
                .05, -.16, f'extremum at: LR = {x_min_str}',
                transform=ax3.transAxes, c='red', alpha=.5)
        ax3.text(
                .05, -.22, f'weighted loss extremum = {x_wmin_str}',
                transform=ax3.transAxes, c='pink')
        if explosion_ratio is not None:
            ax1.set_ylim(
                    bottom=.9*loss_logs[0].min(),
                    top=explosion_ratio*loss_logs[0, 0])
            ax2.set_ylim(
                    bottom=.9*loss_logs[1:].min(),
                    top=explosion_ratio*loss_logs[1:, 0].max())
            ax3.set_ylim(
                    bottom=.9*mllog.min(),
                    top=explosion_ratio*mllog[0])
            ax4.set_ylim(
                    bottom=.9*wllog.min(),
                    top=explosion_ratio*wllog[0])

        fig.tight_layout()
        fig.subplots_adjust(top=0.88);
        print('Minimum loss value (mean):', mllog.mean())

    def trackTrainingScore(self):
        self.meta = 'tranining'
        if self.history[-1] < self.best_score:
            self.best_epoch = self.No
            self.best_score = self.history[-1]
            self.best_weights = copy.deepcopy(self.net.state_dict())

    def validate(self, X, Y):
        self.meta = 'validation'
        current_score = torch.mean((Y - self.net(*X))**2).item()
        if current_score < self.best_score:
            self.best_epoch = self.No
            self.best_score = current_score
            self.best_weights = copy.deepcopy(self.net.state_dict())

    def terminate(self):
        if self.best_weights is not None:
            self.net.load_state_dict(self.best_weights)
        print(f'Best {self.meta} score is {self.best_score}')
        print(f'Achived at epoch #{self.best_epoch}')
