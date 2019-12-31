import torch
from torch import nn

sigma = torch.tanh

class DGMCell(nn.Module):
    def __init__(self, d, M, growing):
        super().__init__()
        self.Uz = nn.Linear(d+1, M, bias=False)
        self.Ug = nn.Linear(d+1, M, bias=False)
        self.Ur = nn.Linear(d+1, M, bias=False)
        self.Uh = nn.Linear(d+1, M, bias=False)

        self.Wz = nn.Linear(M, M)
        self.Wg = nn.Linear(M, M)
        self.Wr = nn.Linear(M, M)
        self.Wh = nn.Linear(M, M)

        self.A = (lambda x: x) if growing else sigma

    def forward(self, SX):
        S, X = SX
        Z = sigma(self.Uz(X) + self.Wz(S))
        G = sigma(self.Ug(X) + self.Wg(S))
        R = sigma(self.Ur(X) + self.Wr(S))
        H = self.A(self.Uh(X) + self.Wh(S*R))
        S = (1-G)*H + Z*S

        return S, X


class ResNetLikeDGM(nn.Module):
    """
    DGM algorithm from https://arxiv.org/pdf/1708.07469.pdf
    Args:
    -----
    d - dimension of the problem
    M - layers' width
    L - recurrency depth
    """
    def __init__(self, d, M=50, L=3, growing=False):
        super().__init__()
        self.W0 = nn.Linear(d+1, M)
        self.W1 = nn.Linear(M, 1)

        self.layers = []
        for l in range(L):
            self.layers.append(DGMCell(d, M, growing))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, *X):
        X = torch.stack(X, -1)
        S = sigma(self.W0(X))
        S,_ = self.layers((S, X))
        return self.W1(S).squeeze_(-1)


class RNNLikeDGM(DGMCell):
    """
    Args:
    -----
    d - dimension of the problem
    M - layers' width
    L - recurrency depth
    """
    def __init__(self, d, M=50, L=3, growing=False):
        super().__init__(d, M, growing)
        self.L = L

        self.W0 = nn.Linear(d+1, M)
        self.W1 = nn.Linear(M, 1)

    def forward(self, *X):
        X = torch.stack(X, -1)
        S = sigma(self.W0(X))
        for l in range(self.L):
            Z = sigma(self.Uz(X) + self.Wz(S))
            G = sigma(self.Ug(X) + self.Wg(S))
            R = sigma(self.Ur(X) + self.Wr(S))
            H = self.A(self.Uh(X) + self.Wh(S*R))
            S = (1-G)*H + Z*S

        return self.W1(S).squeeze_(-1)


class DumbLinear(nn.Module):
    """
    Stack of linear layers with ReLU activations
    """
    def __init__(self, arch='3l', base_width=128):
        super().__init__()
        if arch == '3l': self.main = make_3l(base_width)
        elif arch == '6l': self.main = make_6l(base_width)
        else: raise TypeError("Arg 'arch' must be '3l' or '6l'")

    def forward(self, t, x):
        tx = torch.stack((t, x), -1)
        return self.main(tx).squeeze_(-1)

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
