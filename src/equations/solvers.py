import math
import numpy as np
import scipy.integrate
import torch

from functools import partial
from .utils import *


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


class ThinFoilSolver:
    def __init__(
            self, initial, pulse1, eps, theta,
            pulse2=None, scheme='new',
            xi_lims=None, method='RK45'):
        if pulse2 is None: pulse = pulse1
        else: pulse = pulse1 + pulse2
        self.phi = initial

        def force_vector(xi, w):
            a_y, a_z = pulse(xi)
            u_ps = (a_y-eps*w[1])**2+(a_z-eps*w[2])**2
            E = math.tanh(8*w[0]/theta)

            f0 = .5/w[-1]**2*(1+u_ps-w[-1]**2)
            f1 = 1./w[-1]*(a_y-eps*w[1])
            f2 = 1./w[-1]*(a_z-eps*w[2])
            f3 = eps*(E - u_ps/(1+u_ps))
            return [f0,f1,f2,f3]

        if scheme == 'new':
            if xi_lims is None:
                raise Exception(
                        f"for scheme '{scheme}', "
                        '`xi_lims` must be set')
            self.scheme = 'new'
            self.solve = partial(
                    scipy.integrate.solve_ivp,
                    force_vector, xi_lims, initial, method)
        elif scheme == 'old':
            self.scheme = 'old'
            self.solve = partial(
                    scipy.integrate.odeint,
                        force_vector, initial, tfirst=True)
        else:
            raise Exception('scheme: wrong argument')

    def __call__(self, xi):
        res = self.solve(xi)
        if self.scheme == 'old':
            xyzh = res.T
        else:
            self.state = res
            xyzh = res.y
        return xyzh
