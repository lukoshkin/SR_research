import numpy as np
import torch
import torch.nn as nn


class FlexLayer(nn.Module):
    def __init__(self, images, sigma):
        super().__init__()
        self.images = [None, images]
        self.sigma = sigma

        self.probs = torch.ones(len(images)+1)
        self.probs /= len(self.probs)
        self.probs = nn.Parameter(self.probs)

    def freeze(self):
        id = torch.multinomial(self.probs, 1)
        self.snapshot = self.images[id]
        self.id = id

    def forward(self, X):
        if self.id:
            Y = self.snapshot(X)
            # X = self.upsample(X)
            Y = self.sigma(Y + X)
            return Y
        return X


class GenerativeLayer(nn.Module):
    def __init__(self, images, sigma):
        super().__init__()
        self.images = [None, images]
        self.sigma = sigma

        self.probs = torch.ones(len(images)+1)
        self.probs /= len(self.probs)
        self.probs = nn.Parameter(self.probs)

    def refreeze(self):
        i = torch.multinomial(self.probs, 1)
        self.snapshot = self.images[i]
        if i: self.forward = self._forward
        else: self.forward = lambda x: x

    def _forward(self, X):
        Y = self.snapshot(X)
        # X = self.upsample(X)
        Y = self.sigma(Y + X)
        return Y


class GenerativeModel(nn.Module):
    def __init__(self, images, mask, sigma):
        super().__init__()
        self.images = np.array([*images, None])
        self.sigma = sigma

        self.N, self.M = mask.shape
        self.probs = torch.ones((self.N, self.M))
        self.probs[~mask] = 0

        self.probs /= self.probs.sum(1)[:, None]
        self.probs = nn.Parameter(self.probs)

    def refreeze(self):
        self.ix = torch.multinomial(self.probs, 1).view(-1)
        self.ll = self.probs[torch.arange(self.N), self.ix]
        self.main = np.take(self.images, self.ix)
        self.ll = self.ll.prod()

    def forward(self, X):
        Y = X.clone()
        for i, l in zip(self.ix, self.main):
            if i >= self.M-1: continue
            # ? upsample(X) ?
            Y = self.sigma(l(Y) + X)
        return Y

    def trainBackward(self,):
        pass

    def valBackward(self,):
        pass
