import math
import torch
from torch import optim

import sys
sys.path.append('..')

from dl.DGM import RNNLikeDGM
from equations.pdes import ThinFoilSODE
from equations.solvers import ThinFoilSolver
from train.trainer import SepLossTrainer
from pulse.pulse import Pulse2D
from pulse.utils import pulse_scalar, pulse_torch

# ---> Training Configs
batch_size = 256
num_batches = 100
num_cycles = 200
nsl_steps = 10
nul_steps = 40
lp = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE', device)
# <---

# ---> Problem Setting
a_0y = 20.
a_0z = 20.
xi_0y = 0.
xi_0z = .5*math.pi

tau = 10*math.pi
theta = .02*math.pi
eps = 200*theta

dxi = 1e-3
xi_lims = (0., 20*math.pi)
rbc = np.array([0., 0., 0., 1.])

p_t = Pulse2D(pulse_torch, ((a_0y, a_0z), (xi_0y, xi_0z), tau))
p_s = Pulse2D(pulse_scalar, ((a_0y, a_0z), (xi_0y, xi_0z), tau))
solver = ThinFoilSolver(rbc, p_s, eps, theta)
w = torch.tensor([1./96, 1.], device=device)
# <---

pde = ThinFoilSODE(
        rbc, pulse, theta,
        xi_lims=xi_lims, device=device, solver=solver)
net = RNNLikeDGM(1,4, as_array=False).to(device)
opt = optim.SGD(net.parameters(), 5e-4, weight_decay=5e-7)
sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 500, eta_min=5e-7)

trainer = SepLossTrainer(net, pde, opt, sch)
for i in range(n_cycles):
    for _ in range(nsl_steps):
        trainer.trainOneEpoch(w, num_batches, batch_size, lp)
        trainer.trackTrainingScore()
    trainer.supervised(False)

    for _ in range(nul_steps):
        trainer.trainOneEpoch(w, num_batches, batch_size, lp)
        trainer.trackTrainingScore()
    trainer.supervised(True)

trainer.terminate()
np.savez('../savings/history', trainer.history, trainer.histories())
torch.save(trainer.best_weights, '../savings/weights.pt')
