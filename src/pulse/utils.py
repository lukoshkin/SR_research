import math
import numpy as np

def pulse_scalar(x, A, phi, tau, freq=1):
    if (x < 0) or (x > tau): return 0
    return A*math.sin(math.pi*x/tau)**2 * math.sin(freq*x-tau/2+phi)

def pulse_np(x, A, phi, tau, freq=1):
    out = A*np.sin(math.pi*x/tau)**2 * np.sin(freq*x-tau/2+phi)
    out[x>=tau] = 0
    out[x<0] = 0
    return out

def pulse_torch(x, A, phi, tau, freq=1):
    out = A*torch.sin(math.pi*x/tau)**2 * torch.sin(freq*x-tau/2+phi)
    out[x>=tau] = 0
    out[x<0] = 0
    return out
