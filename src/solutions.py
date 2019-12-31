import math
import numpy as np
import torch

from numba import njit


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
    char = x - a*t
    psi = torch.exp(-alpha*torch.tensor(shift**2))
    res = torch.exp(-alpha*(char-shift)**2)
    out = torch.empty_like(res).fill_(psi.item())
    out[char > 0] = res[char > 0]
    
    return out

def step_function(x, t=0, a=.5, width=.1, shift=.25):
    """
    Exact solution of the problem
    du/dt + a du/dx = 0,
    u(0, x) = heaviside_step_function(x),
    u(t, 0) = heaviside_step_function(0)

    Also, it is used in the work to set initial or boundary conditions in PDEs
    """
    if isinstance(t, torch.Tensor):
        t = t[:, None]
    char = x - a*t
    out = torch.zeros_like(char)
    out[(char>=shift) & (char<=shift+width)] = 1

    return out

def xpdome(x, t=0, a=.5, alpha=200., shift=0.):
    """
    Exact solution of the problem
    du/dt + a du/dx = x,
    u(0, x) = gauss_kernel(x),
    u(t, 0) = gauss_kernel(0)

    Also, it is used in the work to set initial or boundary conditions in PDEs
    """
    if isinstance(t, torch.Tensor):
        t = t[:, None]
    char = x - a*t
    psi = math.exp(-alpha*shift**2) + char**2/(2*a)
    res = torch.exp(-alpha*(char-shift)**2)
    out = torch.empty_like(res)
    out[char < 0] = psi[char < 0]
    out[char >= 0] = res[char >= 0]

    return out + .5*a*t**2 + t*char

def fkdome(x, t=0, a=.5, r=1., alpha=200., shift=0.):
    """
    Exact solution of the problem
    du/dt + a du/dx = ru(1-u),
    u(0, x) = gauss_kernel(x),
    u(t, 0) = gauss_kernel(0)

    Also, it is used in the work to set initial or boundary conditions in PDEs
    """
    # scalar t
    if isinstance(t, torch.Tensor):
        t = t[:, None]
        exp = torch.exp(-r*t)
    else:
        exp = math.exp(-r*t)

    # x is tensor with 0 dim-s
    if not x.dim(): x = x[None]

    char = x - a*t
    rgxi = 1./torch.exp(-alpha*(char-shift)**2)
    out = 1./(1+(rgxi-1)*exp)

    res = torch.empty_like(char)
    rg0 = 1./torch.exp(-alpha*torch.tensor(shift**2))
    res[:] = 1./(1+(rg0-1)*torch.exp(-r*x/a))
    out[char<0] = res[char<0]
    return out


def spike(x, alpha=10., shift=.5):
    """
    Exponential function with the discontinuous derivative at `shift`
    """
    return torch.exp(-alpha*(x-shift).abs())


class FDSolver:
    """
    Base class for explicit finite-difference solver
    """
    def __init__(self, initial, l, dx, device):
        self.x = torch.linspace(0, l, int(l/dx)+1, device=device)
        self.u0 = initial(self.x)

    def _solveKeepingAll(self):
        out = [self.u0]
        for _ in range(self.N):
            out.append(self._oneTimeStep(out[-1]))
        return out

    def _solveKeepingJustLast(self):
        u = self.u0
        for _ in range(self.N):
            u = self._oneTimeStep(u)
        return u

    def solve(self, last_only=False):
        """
        Returns all layers as a list of arrays if `last_only` is `True`
        And just the array of last layer values otherwise
        """
        if last_only:
            return self._solveKeepingJustLast()
        return self._solveKeepingAll()

class AdvectionSolver(FDSolver):
    """
    Solves advection equation using finite-difference method
    (homogeneous equation, however, it can easily be extended)
    """
    def __init__(
            self, initial, a=.5, l=1.,
            T=1., dx=1e-3, device='cpu'):
        super().__init__(initial, l, dx, device)
        self.C = .9
        self.dt = self.C*dx/a
        self.N = int(T/self.dt)

    def buildRHS(self, rhs):
        """
        A vector of length `len(u)-1`
        """
        self._rhs = rhs*self.dt

    def _oneTimeStep(self, u):
        return torch.cat([u[0, None], u[1:]-self.C*(u[1:]-u[:-1])+self._rhs])


class ExplicitFisherSolver(FDSolver):
    """
    Solves Fisher's equation using explicit finite-difference scheme
    """
    def __init__(
            self, initial, a=.1, r=0., l=1., T=1.,
            dx=1e-3, device='cpu'):
        super().__init__(initial, l, dx, device)
        self.C1 = .4
        self.dt = self.C1*dx**2/a
        self.C2 = r*self.dt
        self.N = int(T/self.dt)

    def _oneTimeStep(self, u):
        bar_u = .5 * (u[2:] + u[:-2])
        out = u[1:-1] + 2*self.C1*(bar_u-u[1:-1]) + self.C2*bar_u
        out /= 1 + self.C2*bar_u
        return torch.cat([u[0, None], out, u[-1, None]])


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

class ImplicitFisherSolver(FDSolver):
    """
    Solves Fisher's equation using implicit finite-difference scheme
    (Numpy implementation)
    """
    def __init__(self, u0, a=1e-2, r=4., l=1., T=1., dx=1e-3, dt=1e-3):
        self.u0 = u0
        self.C1 = a*dt/dx/dx
        self.C2 = r*dt
        self.N = int(T/dt)

        M = int(l/dx)+1
        self.L = tridiagCholesky(1+2*self.C1, -self.C1, M-2)

    def _buildRHS(self, u):
        bar_u = .5*(u[2:]+u[:-2])
        rhs = u[1:-1]*(1+self.C2*(1-bar_u))
        rhs[0] += self.C1*u[0]
        rhs[-1] += self.C1*u[-1]
        return rhs

    def _oneTimeStep(self, u):
        rhs = self._buildRHS(u)
        x = solveTridiagCholesky(self.L, rhs)
        return np.r_[u[0], x, u[-1]]


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

    def __X_k(self, x, k):
        return torch.sin(math.pi*k/self.l*x)

    def __T_k(self, t, m):
        """
        k = 2m + 1
        """
        C = self.alpha * self.l
        A_k = 4*(((-1)**m*C + math.pi*(2*m+1)*math.exp(-C/2))
            / (C**2 + (math.pi*(2*m+1))**2)
            - 1./(math.pi*(2*m+1))*math.exp(-C/2))
        
        return A_k * torch.exp(-((2*m+1)*math.pi/self.l)**2 * self.a*t)
    
    def solve(self, t, x, eps=1e-4):
        """
        t       a tensor or a scalar value
        x       must be a tensor to deduce the device
        eps     terms in the series are no less than `eps` (in abs. value)
        """
        m = 0
        if not isinstance(t, torch.Tensor):
            t = x.new(1).fill_(t)
        out = x.new(len(t), len(x)).fill_(0)
        while True:
            coeff = self.__T_k(t, m)
            out += coeff[:, None]*self.__X_k(x, 2*m+1)
            if abs(coeff.max()) < eps:
                break
            m += 1
        return math.exp(-self.alpha*self.l/2) + out

# Algorithmically, the code excerpt below should work faster.
# But in practice, this is not the case for python. I didn't check
# its c++ implementation nor its jit version

# In terms of parallelization, it may not be optimal

#    def solve(self, t, x, eps=1e-4):
#        """
#        t       a tensor or a scalar value
#        x       must be a tensor to deduce the device
#        eps     terms in the series are no less than `eps` (in abs. value)
#        """
#        m = 0
#        if not isinstance(t, torch.Tensor):
#           t = x.new(1).fill_(t)
#        out = x.new(len(t), len(x)).fill_(0)
#        indices = torch.arange(len(t))
#        mask = indices.clone()
#        while indices.nelement():
#            coeff = self.__T_k(t[indices], m)
#            out[indices] += coeff[:, None]*self.__X_k(x, 2*m+1)
#            mask = (abs(coeff) > eps).nonzero().view(-1)
#            indices = indices[mask]
#            m += 1
#        return math.exp(-self.alpha*self.l/2) + out
