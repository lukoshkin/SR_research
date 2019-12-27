import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Anime:
    """
    Args:
    -----
    data        two-tuple: (x, y)
                x is a discretization along X axis
                y[t] is the target values at x points at time t

    text        `text[t]` is info to be printed at frame t
    """
    def __init__(self, fig, data, label=None, string=None, text=None):
        self.line, = plt.plot([], [], lw=2, label=label)
        plt.xlim(data[0].min(), data[0].max())
        self.anime = animation.FuncAnimation(
            fig, self.update, init_func=self.init,
            frames=len(data[1]), interval=20, repeat=False, blit=True)
            # << `repeat=True` leads to the divergence
            #    of graphs if there are more than 2

        self.data = data
        self.string = string
        if string is not None:
            assert text is not None, 'text is not provided'
            self.text = text
            self.ax = plt.gca()

    def init(self):
        self.line.set_data([], [])

    def update(self, t):
        x, y = self.data
        self.line.set_data(x, y[t])
        if self.string is not None:
            self.ax.set_title(self.string.format(self.text[t]))

def running_mean(signal, ww=80):
    """
    Args:
    -----
    ww        window width
    """
    return np.convolve(signal, np.ones(ww)/ww, mode='valid')
