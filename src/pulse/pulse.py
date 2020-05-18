from numbers import Number
from functools import partial

def parasserts(params):
    for i, p in enumerate(params):
        if i < 2: assert len(p) == 2, 'Incorrectly set parameters'
        else: assert isinstance(p, Number), 'Incorrectly set parameters'


class Pulse2D:
    """
    There are two ways to define Pulse:
    1. via generation function and its parameters
    2. by setting pulse components (more general way)
    """
    def __init__(self, gen_fn=None, params=None):
        """
        Parameters
            gen_fn: scalar or vectorized function
            Takes 4 arguments: A - ampl, phi - phase,
            tau - pulse support, freq - frequency

            params: list/tuple
            1. (A, phi, tau, freq) - gen_fn args
            2. (A, phi, tau) - then freq will be set to 1
            3. (p1, p2) - two pulse components
        """
        self.genfn = gen_fn
        self.p = params

    @property
    def p(self):
        return self.__p

    @p.setter
    def p(self, params):
        if params is None:
            self.__p = None
        else:
            if len(params) == 4:
                parasserts(params)
                self._setViaParameters(*params)
            elif len(params) == 2:
                assert all(map(callable, params)), 'Incorrect params'
                self._setViaComponents(*params)
            elif len(params) == 3:
                parasserts(params)
                self._setViaParameters(*params, 1)
            else:
                raise TypeError('params: Too many parameters')

    def _setViaParameters(self, A, phi, tau, freq):
        p0 = partial(self.genfn, A=A[0], phi=phi[0], tau=tau, freq=freq)
        p1 = partial(self.genfn, A=A[1], phi=phi[1], tau=tau, freq=freq)
        self.__p = (p0, p1)

    def _setViaComponents(self, p0, p1):
        self.__p = (p0, p1)

    def __call__(self, x):
        return self.p[0](x), self.p[1](x)

    def __add__(self, another):
        p0 = lambda x: self.p[0](x) + another.p[0](x)
        p1 = lambda x: self.p[1](x) + another.p[1](x)

        p = Pulse2D()
        p.p = (p0, p1)
        return p
