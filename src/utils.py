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


class SepLossTrainer:
    def __init__(self, net, pde, optimizer, scheduler, pbar=False):
        self.No = 0
        self.best_epoch = 0
        self.best_score = float('inf')
        self.best_weights = None
        self.history = []
        self.each_loss_history = []

        self.net = net
        self.pde = pde
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.meta = None
        self._sbatch = None
        self._eps_0, self._eps_r = 1., 0.
        self._iter = lambda n,s: trange(n, desc=s)
        if not pbar: self._iter = lambda n, s: range(n)

    def trainOneEpoch(self, w, num_batches=100, batch_size=128):
        """
        w       weights for the manual balancing of loss terms
        """
        loss_logs = w.new_empty([num_batches, w.nelement()])
        for i in self._iter(num_batches, f'Epoch {self.No}'):
            self.optimizer.zero_grad()

            batch = self.pde.sampleBatch(batch_size)
            L = self.pde.computeLoss(batch, self.net)
            loss_logs[i] = L.data
            (w@L).backward()

            self.optimizer.step()
        self.scheduler.step()
        self.each_loss_history.append(loss_logs.mean(0).cpu().numpy())
        self.history.append(self.each_loss_history[-1].sum())
        self.No += 1

    def linearWarmUp(
            self, w, num_iters, batch_size,
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
        loss_logs = w.new_empty([num_iters, w.nelement()])
        opt_dict = copy.deepcopy(self.optimizer.state_dict())
        if range_test:
            net_weights = copy.deepcopy(self.net.state_dict())

        if lr_lims is not None:
            lr_0, lr_n = lr_lims
            assert lr_n > lr_0, 'inappropriate range'
        else:
            lr_n = self.scheduler.base_lrs[0]
            assert lr_n != 0, 'zero optimizer lr'
            lr_0 = lr_n / num_iters
        xdata = np.linspace(lr_0, lr_n, num_iters, dtype='f4')

        desc = 'range test' if range_test else 'warm-up'
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
                    and explosion_ratio*loss_logs[0].mean() < (w@L).item()):
                print('early stop due to the loss explosion')
                break
        self.optimizer.load_state_dict(opt_dict)
        if range_test:
            self.net.load_state_dict(net_weights)
            fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(9, 4))
            ax2 = ax1.twinx()

            xdata = xdata[:i+1]
            loss_logs = loss_logs.cpu().numpy()[:i+1].T
            x_min = xdata[loss_logs.mean(0).argmin()]
            m, e = f'{x_min:.2e}'.split('e')
            x_min_str = r'${}\times 10^{{{:d}}}$'.format(m, int(e))

            viridis = cm.get_cmap('viridis', len(loss_logs))
            cmap = viridis(np.linspace(0, 1, len(loss_logs)))

            lbls = ['do', 'ic']
            repeats = len(loss_logs) - len(lbls)
            lbls += ['bc'] * repeats
            axes = [ax1, ax2] + [ax2] * repeats

            for lbl, c, ax, loss in zip(lbls, cmap, axes, loss_logs):
                ax.plot(xdata, loss, c=c, alpha=.4, label=lbl)
            lines = ax1.lines + ax2.lines
            fig.suptitle('LR linear range test')
            ax1.set_ylabel('do loss')
            ax2.set_ylabel(','.join(lbls[1:])+' losses')
            ax1.set_xlabel('iteration #')
            ax1.axvline(x_min, 0, 1, c='red', ls='--')

            ax3.plot(xdata, loss_logs.mean(0))
            ax3.set_ylabel('avg loss')
            ax3.axvline(x_min, 0, 1, c='red', ls='--')
            ax3.text(
                    -.05, -.2, f'extremum at: LR = {x_min_str}',
                    transform=ax3.transAxes, c='red', alpha=.5)
            plt.legend(lines, lbls, loc=9)

            fig.tight_layout()
            fig.subplots_adjust(top=0.88);

    def exponentialWarmUp(
            self, w, num_iters, batch_size,
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
            assert lr_n > lr_0 and lr_0 > 0, 'inappropriate range'
            gamma = (lr_n/lr_0)**(1./num_iters)
        else:
            lr_n = self.scheduler.base_lrs[0]
            assert lr_n != 0, 'zero optimizer lr'
            lr_0 = lr_n*1e-5
            gamma = (1e5)**(1./num_iters)

        for par in self.optimizer.param_groups:
            par['lr'] = lr_0
        scheduler = StepLR(self.optimizer, 1, gamma)
        xdata = np.empty(num_iters, 'f4')

        desc = 'range test' if range_test else 'warm-up'
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
                    and explosion_ratio*loss_logs[0].mean() < (w@L).item()): 
                print('early stop due to loss explosion')
                break
        self.optimizer.load_state_dict(opt_dict)
        if range_test:
            self.net.load_state_dict(net_weights)
            fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(9, 4))
            ax2 = ax1.twinx()

            xdata = xdata[:i+1]
            loss_logs = loss_logs.cpu().numpy()[:i+1].T
            x_min = xdata[loss_logs.mean(0).argmin()]
            m, e = f'{x_min:.2e}'.split('e')
            x_min_str = r'${}\times 10^{{{:d}}}$'.format(m, int(e))

            viridis = cm.get_cmap('viridis', len(loss_logs))
            cmap = viridis(np.linspace(0, 1, len(loss_logs)))

            lbls = ['do', 'ic']
            repeats = len(loss_logs) - len(lbls)
            lbls += ['bc'] * repeats
            axes = [ax1, ax2] + [ax2] * repeats

            for lbl, c, ax, loss in zip(lbls, cmap, axes, loss_logs):
                ax.plot(xdata, loss, c=c, alpha=.4, label=lbl)
            lines = ax1.lines + ax2.lines
            fig.suptitle('LR exponential range test')
            ax1.set_ylabel('do loss')
            ax2.set_ylabel(','.join(lbls[1:])+' losses')
            ax1.set_xlabel('iteration #')
            ax1.set_xscale('log')
            ax2.set_xscale('log')

            ax3.plot(xdata, loss_logs.mean(0))
            ax3.set_xscale('log')
            ax3.set_ylabel('avg loss')
            ax3.axvline(x_min, 0, 1, c='red', ls='--')
            ax3.text(
                    -.05, -.2, f'extremum at: LR = {x_min_str}',
                    transform=ax3.transAxes, c='red', alpha=.5)
            plt.legend(lines, lbls, loc=9);

            fig.tight_layout()
            fig.subplots_adjust(top=0.88);

    def trainOneEpochWithRAR(
            self, w, num_batches=100, batch_size=128, k=2):
        """
        w       weights for the manual balancing of loss terms
        k       int, hyperparameter in RAR procedure
        """
        loss_logs = w.new_empty([num_batches, w.nelement()])
        for i in self._iter(num_batches, f'Epoch {self.No}'):
            self.optimizer.zero_grad()
            if self._eps_r < self._eps_0:
                self._sbatch, self._eps_0 = self._makeSupportBatch(
                        batch_size, k)

            batch = self.pde.sampleBatch(batch_size)
            batch_2x = torch.cat([batch, self._sbatch])
            L = self.pde.computeLoss(batch_2x, self.net)
            loss_logs[i] = L.data
            (w@L).backward()

            self.optimizer.step()
            self._eps_r = self.pde.computeLoss(
                    self._sbatch, self.net)[0].item()
        self.scheduler.step()
        self.each_loss_history.append(loss_logs.mean(0).cpu().numpy())
        self.history.append(self.each_loss_history[-1].sum())
        self.No += 1

    def _makeSupportBatch(self, batch_size, k):
        """
        Residual-based Adaptive Refinement
        Section 2.8, https://arxiv.org/pdf/1907.04502.pdf
        ---
        Unfortunately, in the current implementation,
        it is only possible to make differential op-r refinement
        """
        batch = self.pde.sampleBatch(k*batch_size)
        R_do,*_ = self.pde.computeResiduals(batch, self.net)
        values, indices = R_do.detach().sort()
        support_batch = batch[indices[-batch_size:]]
        eps_0 = values[:k//2*batch_size].mean().item()
        return support_batch, eps_0

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
