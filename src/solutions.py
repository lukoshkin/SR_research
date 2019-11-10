import math
import torch

def dome(x, t=0, a=.5, alpha=200., shift=0.):
    """ 
    Exact solution of the problem 
    du/dt + a du/dx = 0, 
    u(0, x) = gauss_kernel(x), 
    u(t, 0) = gauss_kernel(0)

    Also, it is used in the work to set initial or boundary conditions in PDEs
    """
    if isinstance(t, torch.Tensor):
        t = t[:, None]
    psi = torch.exp(-alpha*torch.tensor(shift**2))
    res = torch.exp(-alpha*(x-shift-a*t)**2)
    out = torch.empty_like(res).fill_(psi.item())
    out[x - a*t > 0] = res[x - a*t > 0]
    
    return out


def spike(x, alpha=10., shift=.5):
    """
    Exponential function with discontinuous derivative at `shift`
    """
    return torch.exp(-alpha*(x-shift).abs())


class FDSolver:
    def __init__(self, initial, a, l, T, dx, device):
        self.o = initial(torch.zeros(1, device=device))
        self.x = torch.linspace(0, l, int(l/dx), device=device)
        self.u0 = initial(self.x)

    def __call__(self):
        out = [self.u0]
        for i in range(self.N):
            out.append(self._oneTimeStep(out[-1]))

        return out

class AdvectionSolver(FDSolver):
    """
    Solves advection equation using finite-difference method
    (homogeneous equation, however, it can easily be extended)
    """
    def __init__(
            self, initial, a=.5, l=1., T=1.,
            dx=1e-3, device='cpu'):
        super().__init__(initial, a, l, T, dx, device)
        self.C = -.9
        dt = -self.C*dx/a
        self.N = int(T/dt)

    def _oneTimeStep(self , u):
        return u + torch.cat([self.o, self.C * (u[1:] - u[:-1])])


class FisherSolver(FDSolver):
    """
    Solves Fisher's equation using finite-difference method
    """
    def __init__(
            self, initial, a=.1, r=0., l=1., T=1.,
            dx=1e-3, device='cpu'):
        super().__init__(initial, a, l, T, dx, device)
        self.l = initial(l*torch.ones(1, device=device))
        self.C1 = .4
        dt = self.C1 * dx**2 / a
        self.C2 = r*dt
        self.N = int(T/dt)

    def _oneTimeStep(self, u):
        bar_u = .5 * (u[2:] + u[:-2])
        out = u[1:-1] + 2*self.C1*(bar_u-u[1:-1]) + self.C2*bar_u
        out /= 1 + self.C2*bar_u
        return torch.cat([self.o, out, self.l])


class PartialSolver:
    """ 
    Solves the following problem:  

    u_t + a u_xx = 0, 
    u(0, x) = exp(-alpha |x - l/2|), 
    u(t, 0) = exp(-alpha l/2), 
    u(t, l) = exp(-alpha l/2)

    """
    def __init__(self, a, alpha, l):
        self.a = a
        self.alpha = alpha
        self.l = l

    def X_k(self, x, k):
        return torch.sin(math.pi*k/self.l*x)

    def T_k(self, t, m):
        """
        k = 2m + 1
        """
        C = self.alpha * self.l
        A_k = 4*(((-1)**m*C + math.pi*(2*m+1)*math.exp(-C/2))
            / (C**2 + (math.pi*(2*m+1))**2)
            - 1./(math.pi*(2*m+1))*math.exp(-C/2))
        
        return A_k * torch.exp(-((2*m+1)*math.pi/self.l)**2 * self.a*t)
    
    def __call__(self, t, x, eps=1e-4):
        """
        t       a tensor or a scalar value
        x       must be a tensor to deduce the device
        eps     terms in the series are no less than `eps` (in abs. value)
        """
        m = 0
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float)
        out = x.new(len(t), len(x)).fill_(0)
        while True:
            coeff = self.T_k(t, m)
            out += coeff[:, None]*self.X_k(x, 2*m+1)
            if abs(coeff.max()) < eps:
                break
            m += 1
        return math.exp(-self.alpha*self.l/2) + out

#    def __call__(self, t, x, eps=1e-4):
#        """
#        t       a tensor or a scalar value
#        x       must be a tensor to deduce the device
#        eps     terms in the series are no less than `eps` (in abs. value)
#        """
#        m = 0
#        if not isinstance(t, torch.Tensor):
#            t = torch.tensor([t], dtype=torch.float)
#        out = x.new(len(t), len(x)).fill_(0)
#        indices = torch.arange(len(t))
#        mask = indices.clone()
#        while indices.nelement():
#            coeff = self.T_k(t[indices], m)
#            out[indices] += coeff[:, None]*self.X_k(x, 2*m+1)
#            mask = (abs(coeff) > eps).nonzero().view(-1)
#            indices = indices[mask]
#            m += 1
#        return math.exp(-self.alpha*self.l/2) + out
