import torch
import torch.nn as nn
import torch.autograd as autograd

class PDE2D(nn.Module):
    """
    Base class
    Lu = 0, where L is some spatio-temporal operator
    obeying the boundary conditions
    ----------
    Args:
    -----
    l                 length of calculation area along x axis
                      (starting from 0)
    T                 length of calculation area along t axis
                      (starting from 0)

                      Thus, (0, l) x (0, T) is the training domain
                      
    initial           intial distribution of the function u
    """
    def __init__(self, l, T, initial):
        super().__init__()
        self.l = l
        self.T = T
        self.phi = initial
    
    def sampleBatch(self, N=5120, device='cpu'):
        """
        Simple sampler.  
        Generates pair (t, x), from which one can build the pairs
        (t, x), (t, 0), (0, x)
        Works slightly worse than if these pairs were generated independently
        ----------------------
        N - number of points to sample in the domain (x, t)
        """
        tx = torch.rand(N, 2, device=device)
        tx[:, 0].mul_(self.T)
        tx[:, 1].mul_(self.l)

        return tx
    
    def _D(self, y, x):
        grad = autograd.grad(
            outputs=y, inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True, allow_unused=True)
        
        if len(grad) == 1:
            return grad[0]
        return grad


class AdvectionPDE(PDE2D):
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
    def __init__(self, initial, a=.5, l=1., T=2.):
        super().__init__(l, T, initial)
        self.a = a

    def computeLoss(self, tx, net):
        t, x = torch.unbind(tx, 1)
        o = torch.zeros_like(t)
        u_t, u_x = self._D(net(t,x), (t,x))
        L = (torch.norm(u_t + self.a * u_x)
             + torch.norm(net(o, x) - self.phi(x))
             + torch.norm(net(t, o) - self.phi(x.new(1).fill_(0.))))

        return L
