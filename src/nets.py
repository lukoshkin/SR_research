import torch
from torch import nn

sigma = torch.tanh

class DGM(nn.Module):
    """
    DGM algorithm from https://arxiv.org/pdf/1708.07469.pdf
    Args:
    -----
    d - dimension of the problem
    M - layers' width
    L - recurrency depth
    """
    def __init__(self, d, M=50, L=3):
        super().__init__()
        self.L = L

        self.W0 = nn.Linear(d+1, M)
        self.W1 = nn.Linear(M, 1)

        self.Uz = nn.Linear(d+1, M, bias=False)
        self.Ug = nn.Linear(d+1, M, bias=False)
        self.Ur = nn.Linear(d+1, M, bias=False)
        self.Uh = nn.Linear(d+1, M, bias=False)

        self.Wz = nn.Linear(M, M)
        self.Wg = nn.Linear(M, M)
        self.Wr = nn.Linear(M, M)
        self.Wh = nn.Linear(M, M)

    def forward(self, *X):
        X = torch.stack(X, -1)
        S = sigma(self.W0(X))
        for l in range(self.L):
            Z = sigma(self.Uz(X) + self.Wz(S))
            G = sigma(self.Ug(X) + self.Wg(S))
            R = sigma(self.Ur(X) + self.Wr(S))
            H = sigma(self.Uh(X) + self.Wh(S*R))
            S = (1-G)*H + Z*S

            # remove sigma in the line with H
            # if approximating unbounded/growing functions

        return self.W1(S).squeeze_(-1)

def make_3l(base_width):
    arch_3l =  nn.Sequential(
        nn.Linear(2, base_width*2),
        nn.ReLU(True),
        nn.Linear(base_width*2, base_width*4),
        nn.ReLU(True),
        nn.Linear(base_width*4, 1))
    return arch_3l

def make_6l(base_width):
    arch_6l = nn.Sequential(
        nn.Linear(2, base_width),
        nn.ReLU(True),
        nn.Linear(base_width, base_width*2),
        nn.ReLU(True),
        nn.Linear(base_width*2, base_width*4),
        nn.ReLU(True),
        nn.Linear(base_width*4, base_width*2),
        nn.ReLU(True),
        nn.Linear(base_width*2, base_width),
        nn.ReLU(True),
        nn.Linear(base_width, 1))
    return arch_6l

class DumbLinear(nn.Module):
    def __init__(self, arch='3l', base_width=128):
        super().__init__()
        if arch == '3l': self.main = make_3l(base_width)
        elif arch == '6l': self.main = make_6l(base_width)
        else: raise TypeError("Arg 'arch' must be '3l' or '6l'")
    
    def forward(self, t, x):
        tx = torch.stack((t, x), -1)
        return self.main(tx).squeeze_(-1)
