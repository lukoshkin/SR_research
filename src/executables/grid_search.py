import sys
import math
import numpy as np

sys.path.append('..')
from pulse.pulse import Pulse2D
from pulse.utils import pulse_scalar
from equations.solvers import ThinFoilSolver

### ---> Problem Setting
dxi = 1e-4
xi_lims = (0., 20*np.pi)
n_steps = np.ceil(xi_lims[1]/dxi) + 1
xi = np.linspace(*xi_lims, int(n_steps))

delay = np.pi
a1, a2 = 1., 1.
freq1, freq2 = 1., 1.

A_1y = 20.
phi_1y = 0.
A1 = [A_1y, a1*A_1y]

E = freq1*A_1y**2*(1+a1**2)         # approx. wave energy
A_2y = E/freq2/(1+a2**2)
A2 = [A_2y, a2*A_2y]

phi1 = np.array([phi_1y, phi_1y+.5*np.pi])
phi2 = phi1 - delay

tau = 10*np.pi                      # pulse duration
theta = .02*np.pi                   # thickness
eps = 200*theta

rbc = np.array([0., 0., 0., 1.])
frange = [7., 13.]
### <--- Problem Setting


p1 = Pulse2D(pulse_scalar, (A1, phi1, tau, freq1))
#p2 = Pulse2D(pulse_scalar, (A2, phi2, tau, freq2))
p2 = None

solver = ThinFoilSolver(
        rbc, p1, eps, theta,
        pulse2=p2, scheme='new', xi_lims=xi_lims)

xyzh = solver(xi)


def field(u, gamma, v_p):
    return eps * u / gamma / (1+v_p)


def spectrum_integral(xi, xyzh, frange):
    l, r = frange
    fc = (l+r) / 2
    bw = (r-l) / 2
    x,y,z,h = xyzh

    u_x = np.gradient(x, xi) * h
    u_y = np.gradient(y, xi) * h
    u_z = np.gradient(z, xi) * h

    gamma = np.sqrt(1 + u_x**2 + u_y**2 + u_z**2)
    v_x = u_x / gamma

    t = xi + x
    t_uni = np.linspace(t[0], t[-1], len(xi))

    E_y = field(u_y, gamma, v_x)
    E_z = field(u_z, gamma, v_x)

    E_yi = np.interp(t, t_uni, E_y)
    E_zi = np.interp(t, t_uni, E_z)

    spec_y = np.fft.fft(E_yi)
    spec_z = np.fft.fft(E_zi)

    dt = t_uni[1] - t_uni[0]
    w = np.fft.fftfreq(len(t_uni), d=dt)

    Fl = np.exp(-(w*2*np.pi - fc)**16 / bw**16)
    Fr = np.exp(-(w*2*np.pi + fc)**16 / bw**16)
    F = Fl + Fr

    spf_y = F*spec_y
    spf_z = F*spec_z

    i_upper = math.ceil(dt*len(xi)*(fc+bw)/(2*np.pi))
    i_lower = math.floor(dt*len(xi)*(fc-bw)/(2*np.pi))

    intf_spy = abs(spf_y[i_lower:i_upper]).sum()
    intf_spz = abs(spf_z[i_lower:i_upper]).sum()
    return (intf_spy**2 + intf_spz**2)**.5

print(spectrum_integral(xi, xyzh, frange))
