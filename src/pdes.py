import torch
import torch.nn as nn
import torch.autograd as autograd


class PDEND(nn.Module):
    """
    Base class.
    Lu = 0, where L is some spatio-temporal operator
    obeying the boundary conditions
    ----------
    Args:
    -----
    initial         intial distribution of the function u
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
        self._sigma = (self.lims[:, 1] - self.lims[:, 0]).to(device)
        self._mu = self.lims[:, 0].to(device)

    def sampleBatch(self, N):
        return self._sigma * self._mu.new(
                N, len(self.lims)).uniform_() + self._mu
    
    def _D(self, y, x):
        grad = autograd.grad(
            outputs=y, inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True, allow_unused=True)
        
        if len(grad) == 1:
            return grad[0]
        return grad


class AdvectionPDE(PDEND):
    """
    Homogeneous Advection Equation
    -------------------------------------
    du/dt + a du/dx = 0
    u(0, x) = phi(x)
    u(t, 0) = phi(0)
    -------------------------------------
    Args:
    -----
    a                 wave velocity
    """
    def __init__(self, initial, a=.5, l=1., T=2., device='cpu'):
        super().__init__(initial, device, T, l)
        self.a = a

    def computeLoss(self, tx, net):
        t, x = tx.unbind(1)
        o = torch.zeros_like(t)
        u_t, u_x = self._D(net(t,x), (t,x))
        L = (torch.norm(u_t + self.a * u_x)
             + torch.norm(net(o, x) - self.phi(x))
             + torch.norm(net(t, o) - self.phi(o[0])))

        return L

class FisherPDE(PDEND):
    """
    u_t - a u_xx = r u(1-u)
    u(0, x) = phi(x)
    u(t, 0) = phi(0)
    u(t, l) = phi(l)
    ----------
    Args:
    -----
    a             diffusion coefficient
    r             RHS coefficient
    """
    def __init__(self, initial, a=.1, r=0., l=1., T=1., device='cpu'):
        super().__init__(initial, device, T, l)
        self.a = a
        self.r = r
        self.l = l

    def computeLoss(self, tx, net, delta=.001):
        t, x = torch.unbind(tx, 1)
        o = torch.zeros_like(t)
        l = torch.empty_like(x).fill_(self.l)

        D1 = delta * torch.randn_like(x)
        D2 = delta * torch.randn_like(x)

        y0 = net(t, x)
        u_t, u_x = self._D(y0, (t,x))

        denom = 1./(delta*delta)
        u1_xx_1p = denom * (self._D(net(t, x+D1), x) - u_x) * D1
        u2_xx_1p = denom * (self._D(net(t, x+D2), x) - u_x) * D2
        u1_xx_1m = denom * (self._D(net(t, x-D1), x) - u_x) * D1
        u2_xx_1m = denom * (self._D(net(t, x-D2), x) - u_x) * D2

        G_a1 = (u_t - self.a*u1_xx_1p - self.r*y0*(1-y0))
        G_a2 = (u_t - self.a*u2_xx_1p - self.r*y0*(1-y0))
        G_b1 = (u_t + self.a*u1_xx_1m - self.r*y0*(1-y0))
        G_b2 = (u_t + self.a*u2_xx_1m - self.r*y0*(1-y0))

        L = (torch.mean(G_a1.detach()*G_a2 + G_b1.detach()*G_b2)
             + torch.norm(net(o, x) - self.phi(x))
             + torch.norm(net(t, o) - self.phi(o[0]))
             + torch.norm(net(t, l) - self.phi(l[0])))

        return L

class ParametricFisherPDE(PDEND):
    def __init__(self, initial, a=.1, r=(2, 4), l=1, T=1, device='cpu'):
        super().__init__(initial, device, T, l, r)
        self.a = a
        self.l = l

    def computeLoss(self, txr, net, delta=.001):
        t, x, r = torch.unbind(txr, 1)
        o = torch.zeros_like(t)
        l = torch.empty_like(x).fill_(self.l)

        D1 = delta * torch.randn_like(x)
        D2 = delta * torch.randn_like(x)

        y0 = net(t, x, r)
        u_t, u_x = self._D(y0, (t,x))

        denom = 1./(delta*delta)
        u1_xx_1p = denom * (self._D(net(t, x+D1, r), x) - u_x) * D1
        u2_xx_1p = denom * (self._D(net(t, x+D2, r), x) - u_x) * D2
        u1_xx_1m = denom * (self._D(net(t, x-D1, r), x) - u_x) * D1
        u2_xx_1m = denom * (self._D(net(t, x-D2, r), x) - u_x) * D2

        G_a1 = (u_t - self.a*u1_xx_1p - r*y0*(1-y0))
        G_a2 = (u_t - self.a*u2_xx_1p - r*y0*(1-y0))
        G_b1 = (u_t + self.a*u1_xx_1m - r*y0*(1-y0))
        G_b2 = (u_t + self.a*u2_xx_1m - r*y0*(1-y0))

        L = (torch.mean(G_a1.detach()*G_a2 + G_b1.detach()*G_b2)
             + torch.norm(net(o, x, r) - self.phi(x))
             + torch.norm(net(t, o, r) - self.phi(o[0]))
             + torch.norm(net(t, l, r) - self.phi(l[0])))

        return L
