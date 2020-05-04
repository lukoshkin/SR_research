import copy
import numpy as np
import torch

from functools import partial
from tqdm.autonotebook import trange

import matplotlib.pyplot as plt
from matplotlib import cm


# Evaluation metrics
max_error = lambda u_exact, u_approx: torch.norm(
        u_exact - u_approx, p=float('inf'))
avg_error = lambda u_exact, u_approx: torch.mean(
        (u_exact - u_approx)**2)**.5


def simple_train(
        net, pde, optimizer, scheduler,
        loss_history, num_batches, batch_size=512):

    for _ in trange(num_batches, desc='Training'):
        optimizer.zero_grad()

        batch = pde.sampleBatch(batch_size)
        loss = pde.computeLoss(batch, net)
        loss_history.append(loss.item())
        loss.backward()

        optimizer.step()
        scheduler.step()


class Trainer:
    def __init__(self, net, pde, optimizer, scheduler, pbar=False):
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
        self._sbatch = None
        self._eps_0, self._eps_r = 1., 0.
        self._iter = lambda n,s: trange(n, desc=s)
        if not pbar: self._iter = lambda n, s: range(n)

    def trainOneEpoch(self, num_batches=100, batch_size=512):
        loss_log = []
        for _ in self._iter(num_batches, f'Epoch {self.No}'):
            self.optimizer.zero_grad()

            batch = self.pde.sampleBatch(batch_size)
            loss = self.pde.computeLoss(batch, self.net)
            loss_log.append(loss.item())
            loss.backward()

            self.optimizer.step()
        self.scheduler.step()
        self.history.append(np.mean(loss_log))
        self.No += 1

    def trainOneEpochWithRAR(self, num_batches=100, batch_size=512):
        loss_log = []
        for _ in self._iter(num_batches, f'Epoch {self.No}'):
            self.optimizer.zero_grad()
            if self._eps_r < self._eps_0:
                self._sbatch, self._eps_0 = self._makeSupportBatch(batch_size)

            batch = self.pde.sampleBatch(batch_size)
            batch_2x = torch.cat([batch, self._sbatch])
            loss = self.pde.computeLoss(batch_2x, self.net)
            loss_log.append(loss.item())
            loss.backward()

            self.optimizer.step()
            self._eps_r,_ = self.pde.computeLoss(self._sbatch, self.net)
        self.scheduler.step()
        self.history.append(np.mean(loss_log))
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
