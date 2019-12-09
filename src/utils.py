import numpy as np
import torch
from torch.optim import lr_scheduler

from tqdm import trange
from numba import njit


# Evaluation metrics
max_error = lambda u_exact, u_approx: torch.norm(
        u_exact - u_approx, p=float('inf'))
avg_error = lambda u_exact, u_approx: torch.mean(
        (u_exact - u_approx)**2)**.5


def train(
        net, pde, optimizer, loss_history, num_batches,
        scheduler=None, batch_size=5120):

    if scheduler is None:
        scheduler = lr_scheduler.MultiStepLR(
                optimizer, [1e4, 2e4, 3e4, 4e4], .18)

    for _ in trange(num_batches, desc='Training'):
        optimizer.zero_grad()

        batch = pde.sampleBatch(batch_size)
        batch.requires_grad_(True)

        loss = pde.computeLoss(batch, net)
        loss_history.append(loss.item())
        loss.backward()

        optimizer.step()
        scheduler.step()


@njit
def tridiagCholesky(a, b, n):
    l, m = np.zeros(n), np.zeros(n-1)
    l[0] = a**.5
    for i in range(1, n):
        m[i-1] = b / l[i-1]
        l[i] = (a - m[i-1]**2)**.5

    return l, m

@njit
def solveTridiagCholesky(L, f):
    l, m = L
    n = len(l)

    y = np.empty(n)
    y[0] = f[0]/l[0]
    for i in range(1, n):
        y[i] = (f[i] - m[i-1]*y[i-1]) / l[i]

    x = np.empty(n)
    x[-1] = y[-1]/l[-1]
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - m[i]*x[i+1]) / l[i]

    return x
