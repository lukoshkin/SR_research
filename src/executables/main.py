import math
import torch
from torch import optim

import sys
sys.path.append('..')

from dl.DGM import RNNLikeDGM
from equations.pdes import ThinFoilSODE
from train.trainer import SepLossTrainer


# --> Problem Setting
batch_size = 256
num_batches = 100
num_epochs = 5000
lp = 5

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE', device)

def pulse(x, a, phi, tau):
    out = a * torch.sin(math.pi*x/tau)**2 * torch.sin(x-tau/2+phi)
    out[x>=tau] = 0
    out[x<0] = 0
    return out

w = torch.tensor([1./96, 1.], device=device)
# <--

pde = ThinFoilSODE(
        rbc, pulse, [a_0y, a_0z], [xi_0y, xi_0z],
        tau, theta, xi_lims=xi_lims, device=device)
net = RNNLikeDGM(1,4, as_array=False).to(device)
opt = optim.SGD(net.parameters(), 5e-4, weight_decay=5e-7)
sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 500, eta_min=5e-7)

trainer = SepLossTrainer(net, pde, opt, sch)
for i in range(num_epochs):
    trainer.trainOneEpoch(w, num_batches, batch_size, lp)
    trainer.trackTrainingScore()

trainer.terminate()
np.savez('../savings/history', trainer.history, trainer.histories())
torch.save(trainer.best_weights, '../savings/weights.pt')
