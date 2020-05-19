import math

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
    pulse(A, phi, tau) - function of laser pulse shape.
    It takes as arguments: `A` - pulse amplitude, `tau`
    - pulse duration, and `phi` - pulse phase
    """
    def __init__(
            self, initial, pulse, theta,
            xi_lims=1., device='cpu', solver=None):
        super().__init__(initial, device, xi_lims)
        self.phi = torch.Tensor(initial).to(device)
        self.pulse = pulse
        self.theta = theta
        self.e = 200*theta

        self._solver = solver
        self.supervised(solver is not None)

    def supervised(self, flag):
        if flag:
            assert self._solver is not None, (
            'arg::solver: For supervised learning, a solver is required')
            self.computeResiduals = self._computeResidualsSL
        else: self.computeResiduals = self._computeResidualsUL

    def _computeResidualsUL(self, xi, net):
        xi.squeeze_(-1)
        xi.requires_grad_(True)

        x,y,z,h = net(xi).unbind(1)
        a_y, a_z = self.pulse(xi)
        u_y = a_y - self.e*y
        u_z = a_z - self.e*z
        u_ps = u_y**2 + u_z**2

        R_bc = net(xi.new([0.])) - self.phi
        R1 = 2*D(x,xi)*h**2 - (1+u_ps-h**2)
        R1 = R1 / torch.norm(h**2).item()

        R2 = h*D(y,xi) - u_y
        R2 = R2 / torch.norm(h).item()
        R3 = h*D(z,xi) - u_z
        R3 = R3 / torch.norm(h).item()

        R4 = D(h,xi) - self.e*(torch.tanh(8*x/self.theta)-u_ps/(1+u_ps))
        R_do = torch.stack([R1, R2, R3, R4])

        xi.requires_grad_(False)
        return torch.norm(R_do, dim=0), R_bc.abs()

    def _computeResidualsSL(self, xi, net):
        """
        This is an experimental part (pretty slow).
        Most likely, if is not reworked, it will be removed in the future
        """
        xi.squeeze_(-1)
        Y_pred = net(xi)
        xi_sorted = torch.sort(xi)[0].cpu().numpy()
        V = self._solver(xi_sorted)
        Y = Y_pred.new(Y_pred.shape)
        Y[:] = torch.tensor(V)

        R_do = Y - Y_pred
        R_bc = net(xi.new([0.])) - self.phi
        return torch.norm(R_do, dim=1), R_bc.abs()

    def computeLoss(self, xi, net):
        R_do, R_bc = self.computeResiduals(xi, net)
        return torch.stack([R_do.mean(), R_bc.sum()])
