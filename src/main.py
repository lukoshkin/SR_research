import math
import scipy.integrate

import torch
from torch import optim
import matplotlib.pyplot as plt

from nets import *
from pdes import *
from utils import *
from graphics import *
from solutions import *

# --> Problem Setting
batch_size = 600
num_batches = 100
num_epochs = 1000

a_0y = 20.
a_0z = 20.
xi_0y = 0.
xi_0z = .5*math.pi

tau = 10*math.pi
theta = .02*math.pi
eps = 200*theta

dxi = 1e-3
xi_lims = (0., 1.)
rbc = np.array([0., 0., 0., 1.])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pulse(x, a, phi, tau):
    out = a * torch.sin(math.pi*x/tau)**2 * torch.sin(x-tau/2+phi)
    out[x>=tau] = 0
    out[x<0] = 0
    return out
# <--

pde = ThinFoilSPDE(
        rbc, pulse, [a_0y,a_0z], [xi_0y,xi_0z],
        tau, theta, xi_lims=xi_lims, device=device)
net = RNNLikeDGM(1,4,200,5, as_array=True).to(device)
opt = optim.SGD(net.parameters(), 1e-3, weight_decay=3e-5)
sch = optim.lr_scheduler.StepLR(opt, 1, .98)

trainer = SepLossTrainer(net, pde, opt, sch, pbar=True)
for i in range(num_epochs):
    trainer.trainOneEpoch(num_batches, batch_size)
    trainer.trackTrainingScore()

plt.figure(figsize=(18, 6))
gs = plt.GridSpec(2, 2, wspace=.15, hspace=.4)

plt.subplot(gs[0:, 0])
plt.plot(trainer.history)
plt.xlabel('i', size=20)
plt.ylabel('r', size=20)
plt.title('loss history', size=15)

plt.subplot(gs[0, 1])
plt.plot(trainer.do_loss_history);
plt.title('diff. op. residuals dynamics', size=15)
plt.xticks([])

plt.subplot(gs[1, 1])
plt.plot(trainer.bc_loss_history)
plt.title('bc residuals dynamics', size=15)
plt.xlabel('i', size=20)

plt.savefig('loss.png')

# Numerical solution to system of PDEs describing
# the model of interation of thin foil with high-intense laser pulse. 
# 
# The restoring field (which is defined via sign funciton)
# is smoothed using the `tanh`
# 
# The notation used here:
# xi = t-x
# w[0] -> x(xi)
# w[1] -> y(xi)
# w[2] -> z(xi)
# w[3] -> h(xi)

def pulse_scalar(x, A, phi):
    if (x < 0) or (x>tau): return 0.
    return A*math.sin(math.pi*x/tau)**2 * math.sin(x-tau/2+phi)

def force_vector(xi, w):
    a_y = pulse_scalar(xi, a_0y, xi_0y)
    a_z = pulse_scalar(xi, a_0z, xi_0z)
    u_ps = (a_y-eps*w[1])**2+(a_z-eps*w[2])**2
    E = math.tanh(8*w[0]/theta)

    f0 = .5/w[-1]**2*(1+u_ps-w[-1]**2)
    f1 = 1./w[-1]*(a_y-eps*w[1])
    f2 = 1./w[-1]*(a_z-eps*w[2])
    f3 = eps*(E - u_ps/(1+u_ps))
    return [f0,f1,f2,f3]

r=scipy.integrate.ode(force_vector, jac=None)
r.set_integrator('dopri5')
r.set_initial_value(rbc, xi_lims[0])

w = rbc.copy()
xi = [xi_lims[0]]
while r.successful() and r.t < xi_lims[1]:
    r.integrate(r.t+dxi)
    w = np.vstack([w, r.y])
    xi = np.r_[xi, r.t]

plt.figure()
plt.plot(xi, w[:, 0], label='numerical solution')

xi = torch.Tensor(xi, device=device)
xi.requires_grad_(True)
xyzh = net(xi.view(-1, 1)).unbind(1)
x = xyzh[0].detach().cpu()

plt.plot(xi.detach().cpu(), x, label="net's solution")

plt.xlabel(r'$\xi$')
plt.ylabel(r'$x$')
plt.legend()
plt.savefig('sols.png')

def calc_point_residuals(xi, x, y, z, h):
    a_y = pulse(xi, a_0y, xi_0y, tau)
    a_z = pulse(xi, a_0z, xi_0z, tau)

    u_y = a_y-eps*y
    u_z = a_z-eps*z
    u_ps = u_y**2 + u_z**2

    out = ((D(x,xi)-.5*(1+u_ps-h**2)/h**2).abs()
           + (D(y,xi)-u_y/h).abs()
           + (D(z,xi)-u_z/h).abs()
           + (D(h,xi)-eps*(torch.tanh(8*x/theta)-u_ps/(1+u_ps))).abs())
    return out

plt.figure()
res = calc_point_residuals(xi, *xyzh)
plt.plot(xi.detach().cpu(), res.detach().cpu())
plt.xlabel(r'$\xi$')
plt.ylabel('err')
plt.savefig('pw_loss.png')
