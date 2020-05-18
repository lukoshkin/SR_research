import numpy as np


def running_mean(signal, ww=80):
    """
    Args:
    -----
    ww        window width
    """
    return np.convolve(signal, np.ones(ww)/ww, mode='valid')


def minstrX(xdata, ydata):
    """
    xdata: numpy.ndarray
    ydata: numpy.ndarray
    """
    x_min = xdata[ydata.argmin()]
    m, e = f'{x_min:.2e}'.split('e')
    x_min_str = r'${}\times 10^{{{:d}}}$'.format(m, int(e))
    return x_min_str, x_min
