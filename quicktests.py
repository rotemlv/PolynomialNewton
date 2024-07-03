from poly import *
from newton_raphson import *
import matplotlib.pyplot as plt


def useless_test():
    pol = Polynomial([1, -1]) * Polynomial([1, -2]) * Polynomial([1, -3])
    der = pol.get_derivative_as_object()
    print(der)

    x = np.linspace(1, 3, 1000)

    plt.plot(x, pol(x))
    plt.show()
    print(der.get_derivative_as_object())
    der_roots = [newtons_method(1.5, der, der.get_derivative_as_object(), 1e-7, 1e-7, 1000),
                 newtons_method(3, der, der.get_derivative_as_object(), 1e-7, 1e-7, 1000)]

    print(der_roots)

    pol_roots = [newtons_method(der_roots[0] - 1, pol, pol.get_derivative_as_object(), 1e-7, 1e-7, 1000),
                 newtons_method(der_roots[-1] + 1, pol, pol.get_derivative_as_object(), 1e-7, 1e-7, 1000),
                 newtons_method((der_roots[0] + der_roots[1]) / 2, pol, pol.get_derivative_as_object(), 1e-7, 1e-7,
                                1000)]

    print(pol_roots)
