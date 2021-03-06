import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as WN

sigma = torch.tanh

class DGMCell(nn.Module):
    def __init__(self, d, M, growing, weight_norm):
        super().__init__()
        wn = WN if weight_norm else lambda x: x

        self.Uz = wn(nn.Linear(d, M, bias=False))
        self.Ug = wn(nn.Linear(d, M, bias=False))
        self.Ur = wn(nn.Linear(d, M, bias=False))
        self.Uh = wn(nn.Linear(d, M, bias=False))

        self.Wz = wn(nn.Linear(M, M))
        self.Wg = wn(nn.Linear(M, M))
        self.Wr = wn(nn.Linear(M, M))
        self.Wh = wn(nn.Linear(M, M))

        self.A = (lambda x: x) if growing else sigma

    def forward(self, SX):
        S, X = SX
        Z = sigma(self.Uz(X) + self.Wz(S))
        G = sigma(self.Ug(X) + self.Wg(S))
        R = sigma(self.Ur(X) + self.Wr(S))
        H = self.A(self.Uh(X) + self.Wh(S*R))
        S = (1-G)*H + Z*S

        return S, X


def _set_convert(flag):
    if flag: return lambda X: X[0]
    return lambda X: torch.stack(X, -1)


class ResNetLikeDGM(nn.Module):
    """
    DGM algorithm from https://arxiv.org/pdf/1708.07469.pdf
    Args:
    -----
    d_in and d_out- input and ouput dimensions of the problem
    M - layers' width
    L - recurrency depth
    """
    def __init__(
            self, d_in, d_out, M=50, L=3,
            growing=False, as_array=True, weight_norm=False):
        super().__init__()
        wn = WN if weight_norm else lambda x: x
        self.W0 = wn(nn.Linear(d_in, M))
        self.W1 = wn(nn.Linear(M, d_out))
        self._convert = _set_convert(as_array)

        self.layers = []
        for l in range(L):
            self.layers.append(DGMCell(d_in, M, growing, weight_norm))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, *X):
        X = self._convert(X)
        S = sigma(self.W0(X))
        S,_ = self.layers((S, X))
        return self.W1(S).squeeze_(-1)


class RNNLikeDGM(DGMCell):
    """
    Args:
    -----
    d_in and d_out- input and ouput dimensions of the problem
    M - layers' width
    L - recurrency depth
    """
    def __init__(
            self, d_in, d_out, M=50, L=3,
            growing=False, as_array=True, weight_norm=False):
        super().__init__(d_in, M, growing, weight_norm)
        self.L = L

        wn = WN if weight_norm else lambda x: x
        self.W0 = wn(nn.Linear(d_in, M))
        self.W1 = wn(nn.Linear(M, d_out))
        self._convert = _set_convert(as_array)

    def forward(self, *X):
        X = self._convert(X)
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
