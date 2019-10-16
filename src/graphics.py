import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Anime:
    """
    Args:
    -----
    data        two-tuple: (x, y)
                x is a discretization along X axis
                y[t] is the target values at x points
                at time t
    """
    def __init__(self, fig, data, label=None):
        self.line, = plt.plot([], [], lw=2, label=label)
        plt.xlim(data[0].min(), data[0].max())
        self.anime = animation.FuncAnimation(
            fig, self.update, init_func=self.init,
            frames=400, interval=20, repeat=False, blit=True)
    # << `repeat=True` leads to the divergence 
    #    of graphs if there are more than 2

        self.data = data

    def init(self):
        self.line.set_data([], [])

    def update(self, t):
        x, y = self.data
        self.line.set_data(x, y[t])

def running_mean(signal, ww=80):
    """
    Args:
    -----
    ww        window width
    """
    return np.convolve(signal, np.ones(ww)/ww, mode='valid')
