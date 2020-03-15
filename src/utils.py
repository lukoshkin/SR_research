import copy
import numpy as np
import torch

from functools import partial
from tqdm import trange


# Evaluation metrics
max_error = lambda u_exact, u_approx: torch.norm(
        u_exact - u_approx, p=float('inf'))
avg_error = lambda u_exact, u_approx: torch.mean(
        (u_exact - u_approx)**2)**.5


def train(
        net, pde, optimizer, scheduler,
        loss_history, num_batches, batch_size=512):

    for _ in trange(num_batches, desc='Training'):
        optimizer.zero_grad()

        batch = pde.sampleBatch(batch_size)
        batch.requires_grad_(True)

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

        self._iter = partial(trange, desc=f'Epoch {self.No}')
        if not pbar: self._iter = range

    def trainOneEpoch(self, num_batches=100, batch_size=512):
        loss_log = []
        for _ in self._iter(num_batches):
            self.optimizer.zero_grad()

            batch = self.pde.sampleBatch(batch_size)
            batch.requires_grad_(True)

            loss = self.pde.computeLoss(batch, self.net)
            loss_log.append(loss.item())
            loss.backward()

            self.optimizer.step()
        self.scheduler.step()
        self.history.append(np.mean(loss_log))
        self.No += 1

    def trackTrainScore(self):
        if self.history[-1] < self.best_score:
            self.best_epoch = self.No
            self.best_score = self.history[-1]
            self.best_weights = copy.deepcopy(self.net.state_dict())

    def validate(self, X, Y):
        current_score = torch.mean((Y - self.net(*X))**2).item()
        if current_score < self.best_score:
            self.best_epoch = self.No
            self.best_score = current_score
            self.best_weights = copy.deepcopy(self.net.state_dict())

    def terminate(self):
        if self.best_weights is not None:
            self.net.load_state_dict(self.best_weights)
        print(f'Best validation score is {self.best_score}')
        print(f'Achived at epoch #{self.best_epoch}')
