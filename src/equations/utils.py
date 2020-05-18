import numpy as np
from numba import njit


@njit
def tridiagCholesky(a, b, n):
    l, m = np.zeros(n), np.zeros(n-1)
    l[0] = a**.5
    for i in range(1, n):
        m[i-1] = b / l[i-1]
        l[i] = (a - m[i-1]**2)**.5

    return l, m


@njit
def solveTridiagCholesky(L, f):
    l, m = L
    n = len(l)

    y = np.empty(n)
    y[0] = f[0]/l[0]
    for i in range(1, n):
        y[i] = (f[i] - m[i-1]*y[i-1]) / l[i]

    x = np.empty(n)
    x[-1] = y[-1]/l[-1]
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - m[i]*x[i+1]) / l[i]

    return x
