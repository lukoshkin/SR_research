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
