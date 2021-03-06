import math
from functools import partial

import torch
import torch.nn as nn
import torch.autograd as autograd


def D(y, x):
    grad = autograd.grad(
        outputs=y, inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True, allow_unused=True)

    if len(grad) == 1:
        return grad[0]
    return grad


class PDE(nn.Module):
    """
    Base class.
    Lu = 0, where L is some spatio-temporal operator
    obeying the boundary conditions
    ----------
    Args:
    -----
    initial         intial distribution of the function u at some time t0
    lims            limits specifying the training domain.
                    It should be pairs `(a,b)` or just right
                    endpoint `b`

    which_sampler   sampling technique to use
    """
    def __init__(self, initial, device, *lims):
        super().__init__()
        self.phi = initial

        lims = list(lims)
        for i, l in enumerate(lims):
            if (not isinstance(l, tuple)
                    and not isinstance(l, list)):
                assert l > 0, 'got incorrect value'
                lims[i] = (0, l)
            else:
                a, b = l
                assert b > a, 'got incorrect value'
        self.lims = torch.tensor(lims, dtype=torch.float)
        self._len = (self.lims[:, 1] - self.lims[:, 0]).to(device)
        self._shift = self.lims[:, 0].to(device)

    def sampleBatch(self, N):
        """
        sample batch of shape (N, d),
        where d is the number of dimensions in the PDE
        """
        return self._len * self._len.new(
                N, len(self.lims)).uniform_() + self._shift


class AdvectionPDE(PDE):
    """
    Homogeneous Advection Equation
    -------------------------------------
    du/dt + a du/dx = f(t,x,u)
    u(t_0,x) = phi(x)
    u(t,x_0) = phi(0)
    -------------------------------------
    Args:
    -----
    a               wave velocity

    rhs             additional term in the diff. equation
                    that can be written as the source term.
                    It takes 3 positional arguments: t, x, u

    r               Damping factor for balancing diff. operator
                    and BCs approximation losses. It should be
                    already accounted in rhs
    """
    def __init__(
            self, initial, a=.5, l=1., T=2., r=1.,
            rhs=lambda t,x,u: 0, device='cpu'):
        super().__init__(initial, device, T, l)
        self.a = a
        self.r = r
        self.rhs = rhs

    def computeLoss(self, tx, net):
        tx.requires_grad_(True)
        t, x = tx.unbind(1)
        t0 = torch.empty_like(t).fill_(self.lims[0, 0])
        x0 = torch.empty_like(t).fill_(self.lims[1, 0])

        u = net(t, x)
        u_t, u_x = D(u, (t,x))
        L = (torch.norm(self.r*(u_t+self.a*u_x) - self.rhs(t, x, u))
             + torch.norm(net(t0, x) - self.phi(x))
             + torch.norm(net(t, x0) - self.phi(x0[0])))

        tx.requires_grad_(False)
        return L


class FisherPDE(PDE):
    """
    u_t - a u_xx = r u(1-u)
    u(t_0,x) = phi(x)
    u(t,x_0) = phi(0)
    u(t,l) = phi(l)
    ----------
    Args:
    -----
    a             diffusion coefficient
    r             RHS coefficient (one fixed value)
    """
    def __init__(
            self, initial, a=.1, r=0., l=1., T=1.,
            device='cpu', useAV=False):
        super().__init__(initial, device, T, l)
        self.a = a
        self.l = l
        self.r = r

        self._d = 1./abs(r) if abs(r) > 1. else 1.
        self.computeLoss = self._lossWithAV if useAV else self._casualLoss

    def _casualLoss(self, tx, net):
        tx.requires_grad_(True)
        t, x = torch.unbind(tx, 1)
        t0 = torch.empty_like(t).fill_(self.lims[0, 0])
        x0 = torch.empty_like(t).fill_(self.lims[1, 0])
        l = torch.empty_like(x).fill_(self.l)

        u = net(t, x)
        u_t, u_x = D(u, (t,x))
        u_xx = D(u_x, x)

        diff_op_loss = self._d*(u_t - self.a*u_xx - self.r*u*(1-u))
        L = (torch.norm(diff_op_loss)
             + torch.norm(net(t0, x) - self.phi(x))
             + torch.norm(net(t, x0) - self.phi(x0[0]))
             + torch.norm(net(t, l) - self.phi(l[0])))

        tx.requires_grad_(False)
        return L

    def _lossWithAV(self, tx, net, delta=1e-3):
        tx.requires_grad_(True)
        t, x = torch.unbind(tx, 1)
        t0 = torch.empty_like(t).fill_(self.lims[0, 0])
        x0 = torch.empty_like(t).fill_(self.lims[1, 0])
        l = torch.empty_like(x).fill_(self.l)

        D1 = delta * torch.randn_like(x)
        D2 = delta * torch.randn_like(x)

        y0 = net(t, x)
        u_t, u_x = D(y0, (t,x))

        denom = 1./(delta*delta)
        u1_xx_1p = denom * (D(net(t, x+D1), x) - u_x) * D1
        u2_xx_1p = denom * (D(net(t, x+D2), x) - u_x) * D2
        u1_xx_1m = denom * (D(net(t, x-D1), x) - u_x) * D1
        u2_xx_1m = denom * (D(net(t, x-D2), x) - u_x) * D2

        G_a1 = self._d*(u_t - self.a*u1_xx_1p - self.r*y0*(1-y0))
        G_a2 = self._d*(u_t - self.a*u2_xx_1p - self.r*y0*(1-y0))
        G_b1 = self._d*(u_t + self.a*u1_xx_1m - self.r*y0*(1-y0))
        G_b2 = self._d*(u_t + self.a*u2_xx_1m - self.r*y0*(1-y0))

        L = (torch.mean(G_a1.detach()*G_a2 + G_b1.detach()*G_b2)
             + torch.norm(net(t0, x) - self.phi(x))
             + torch.norm(net(t, x0) - self.phi(x0[0]))
             + torch.norm(net(t, l) - self.phi(l[0])))

        tx.requires_grad_(False)
        return L


class ParametricFisherPDE(PDE):
    """
    Solves F-K equation in the given interval of parameters
    -----
    Args:
    -----
    r           RHS coefficient (region to explore)
    """
    def __init__(self, initial, a=.1, r=(2, 4), l=1, T=1, device='cpu'):
        super().__init__(initial, device, T, l, r)
        self.a = a
        self.l = l

    def computeLoss(self, txr, net, delta=1e-3):
        txr.requires_grad_(True)
        t, x, r = torch.unbind(txr, 1)
        t0 = torch.empty_like(t).fill_(self.lims[0, 0])
        x0 = torch.empty_like(t).fill_(self.lims[1, 0])
        l = torch.empty_like(x).fill_(self.l)

        D1 = delta * torch.randn_like(x)
        D2 = delta * torch.randn_like(x)

        y0 = net(t, x, r)
        u_t, u_x = D(y0, (t,x))

        denom = 1./(delta*delta)
        u1_xx_1p = denom * (D(net(t, x+D1, r), x) - u_x) * D1
        u2_xx_1p = denom * (D(net(t, x+D2, r), x) - u_x) * D2
        u1_xx_1m = denom * (D(net(t, x-D1, r), x) - u_x) * D1
        u2_xx_1m = denom * (D(net(t, x-D2, r), x) - u_x) * D2

        G_a1 = (u_t - self.a*u1_xx_1p - r*y0*(1-y0))
        G_a2 = (u_t - self.a*u2_xx_1p - r*y0*(1-y0))
        G_b1 = (u_t + self.a*u1_xx_1m - r*y0*(1-y0))
        G_b2 = (u_t + self.a*u2_xx_1m - r*y0*(1-y0))

        L = (torch.mean(G_a1.detach()*G_a2 + G_b1.detach()*G_b2)
             + torch.norm(net(t0, x, r) - self.phi(x))
             + torch.norm(net(t, x0, r) - self.phi(x0[0]))
             + torch.norm(net(t, l, r) - self.phi(l[0])))

        tx.requires_grad_(False)
        return L
