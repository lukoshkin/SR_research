import math
from functools import partial

import torch
import torch.nn as nn
import torch.autograd as autograd

from obsolete.pdes import D, PDE


class UpdatedAdvectionPDE(PDE):
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
    """
    def __init__(
            self, initial, a=.5, l=1., T=2.,
            rhs=lambda t,x,u: 0, device='cpu'):
        super().__init__(initial, device, T, l)
        self.a = a
        self.rhs = rhs

    def computeResiduals(self, tx, net):
        tx.requires_grad_(True)
        t, x = tx.unbind(1)
        t0 = torch.empty_like(t).fill_(self.lims[0, 0])
        x0 = torch.empty_like(t).fill_(self.lims[1, 0])

        u = net(t, x)
        u_t, u_x = D(u, (t,x))
        R_do = u_t + self.a*u_x - self.rhs(t, x, u)
        R_ic = net(t0, x) - self.phi(x)
        R_bc = net(t, x0) - self.phi(x0[0])
        
        tx.requires_grad_(False)
        return torch.stack([R_do.abs(), R_ic.abs(), R_bc.abs()])

    def computeGradNorms(self, tx, net):
        tx.requires_grad_(True)
        t, x = tx.unbind(1)

        u = net(t, x)
        u_t, u_x = D(u, (t,x))
        grad_norms = u_t.abs() + (self.a*u_x).abs()

        tx.requires_grad_(False)
        return grad_norms

    def computeLoss(self, tx, net):
        R = self.computeResiduals(tx, net)
        return R.mean(1)


class ThinFoilSODE(PDE):
    """
    Solves system of PDEs describing the model of interaction
    of thin foil with high-intense laser pulse
    ---
    Args:
    ---
    pulse(a, phi, tau) - function of laser pulse shape.
    It takes as arguments: `a` - pulse amplitude, `tau`
    - pulse duration, and `phi` - pulse phase
    """
    def __init__(
            self, initial, pulse, A0, xi_0,
            tau, theta, xi_lims=1., device='cpu'):
        super().__init__(initial, device, xi_lims)
        self.phi = torch.Tensor(initial).to(device)
        self.Py = partial(pulse, a=A0[0], phi=xi_0[0], tau=tau)
        self.Pz = partial(pulse, a=A0[1], phi=xi_0[1], tau=tau)
        self.theta = theta
        self.e = 200*theta

    def computeResiduals(self, xi, net):
        xi.squeeze_(-1)
        xi.requires_grad_(True)

        x,y,z,h = net(xi).unbind(1)
        u_y = self.Py(xi) - self.e*y
        u_z = self.Pz(xi) - self.e*z
        u_ps = u_y**2 + u_z**2

        R_bc = net(xi.new([0.])) - self.phi
        R1 = 2*D(x,xi)*h**2 - (1+u_ps-h**2)
        R2 = h*D(y,xi) - u_y
        R3 = h*D(z,xi) - u_z
        R4 = D(h,xi) - self.e*(torch.tanh(8*x/self.theta)-u_ps/(1+u_ps))
        R_do = torch.stack([R1, R2, R3, R4])

        xi.requires_grad_(False)
        return torch.norm(R_do, dim=0), R_bc.abs()

    def computeLoss(self, xi, net):
        R_do, R_bc = self.computeResiduals(xi, net)
        return torch.stack([R_do.mean(), R_bc.sum()])
