from __future__ import annotations
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

USE_CONV = False


class Polynomial:
    def __init__(self, coeffs: None | np.ndarray | list, rank=None, is_dummy=False):
        if coeffs is None:
            coeffs = np.array([0], dtype=np.float64)
        if rank is None:
            rank = len(coeffs) - 1
        assert len(coeffs) == rank + 1, "Improper object instantiation, rank must be equal to len(coeffs) - 1"
        if isinstance(coeffs, list):
            coeffs = np.array(coeffs, dtype=np.float64)
        self.coeffs = coeffs
        self.rank = rank
        self.poly_size = len(self.coeffs)
        self.derivative = None
        # Newton poly is the numerator for the newton algorithm after optimizing for the division operation
        # in essence, it is the inverse polynomial times x. Calculated in place and stored for future calcs
        if not is_dummy:
            self.inverse_polynomial = Polynomial(self.coeffs[::-1], self.rank, True)

    def __call__(self, *args, **kwargs):
        eval_at = args[0]
        assert len(args) == 1 and isinstance(eval_at,
                                             (int, float, np.ndarray, np.number, np.integer))
        # print(f"Evaluating polynomial {self} at {eval_at}")
        if isinstance(eval_at, np.ndarray):
            # Evaluate many points at the same time (good for matplotlib plotting)
            return self.eval_many(eval_at)
        return self.poly_eval(eval_at)

    def __mul__(self, other):
        if USE_CONV:
            # This is much faster
            return self.mul_conv(other)
        m = len(other.coeffs)
        a = np.r_[self.coeffs, np.zeros(m)]
        b = np.tile(a, m)[:-m].reshape(m, -1).T
        return Polynomial(b @ other.coeffs.T, self.rank + other.rank)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        st = ""
        t_rank = self.rank
        for c1, c2 in zip(self.coeffs[:-1], self.coeffs[1:]):
            if c1 != 0:
                st += (f"{str(c1) if c1!=1 else ''}x^{t_rank}" + (" +" if c2 > 0 else " "))
            t_rank -= 1
        st += f"{self.coeffs[-1]}"
        return st

    def mul_conv(self, other):
        return Polynomial(np.convolve(self.coeffs, other.coeffs), self.rank + other.rank)

    def poly_eval(self, at):
        x_pows = at * np.ones(self.poly_size)
        x_pows[0] = 1
        x_pows = np.multiply.accumulate(x_pows)
        return np.dot(x_pows[::-1], self.coeffs)

    def eval_many(self, at):
        x_pows = np.multiply(at.reshape(-1, 1), np.tile(np.ones(self.poly_size), (len(at), 1)))
        x_pows[:, 0] = 1
        x_pows = np.multiply.accumulate(x_pows, axis=1)
        return np.dot(np.flip(x_pows, axis=1), self.coeffs)
        # return np.dot(x_pows, self.coeffs[::-1])

    def get_derivative_as_object(self):
        if self.derivative is not None:
            return Polynomial(self.derivative, self.rank - 1)
        self.derivative = (np.arange(len(self.coeffs) - 1, -1, -1) * self.coeffs)[:-1]
        return Polynomial(self.derivative, self.rank - 1)

    def division_eval(self, denominator: Polynomial, at):
        """Evaluate p(x) / q(x) when both ranks are similar (same or about the same)."""
        if abs(at) <= 1:
            return self(at) / denominator(at)
        if self.rank == denominator.rank:
            return self.inverse_polynomial(1 / at) / denominator.inverse_polynomial(1 / at)
        if self.rank == denominator.rank + 1:
            # return self.inversepoly * x (1/at) / other.inversepoly(1/at)
            tmp = at * self.inverse_polynomial(1 / at)
            # print(f"Enumerator = {tmp}")
            return tmp / denominator.inverse_polynomial(1 / at)
        raise NotImplementedError("Yeah yeah")

    def cauchy_bound(self):
        """A very quick-and-dirty method to find the bound for the absolute values of the solutions of a polynomial.
        Other, better bounds do exist, but require more care"""
        assert len(self.coeffs) and self.coeffs[0] != 0, "Cannot calculate Cauchy's bound in case a_n is zero"
        a_n = self.coeffs[0]
        return 1 + np.max(np.abs(self.coeffs[1:] / a_n))

    def fujiwara_bound(self):
        """A slightly better bound."""
        assert len(self.coeffs) and self.coeffs[0] != 0, "Cannot calculate Fujiwara's bound in case a_n is zero"
        a_n = self.coeffs[0]
        ones = np.ones(self.rank, dtype=self.coeffs.dtype)
        pows = ones / np.arange(1, self.poly_size)
        b = np.abs(self.coeffs[1:] / a_n)
        b[-1] /= 2
        return 2 * np.max(np.pow(b, pows))

    def get_absolute_root_bound(self):
        """Returns the tighter bound out of Cauchy and Fujiwara"""
        a, b = self.cauchy_bound(), self.fujiwara_bound()
        # print(f"Cauchy bound = {a}, Fujiwara bound = {b}")
        return min(a, b)

    def get_normalized_poly(self, normalize_by=None):
        """Return a normalized polynomial object, with all coefficients divided by the maximum absolute value thereof"""
        if normalize_by is None:
            return Polynomial(self.coeffs / np.max(np.abs(self.coeffs)))


if __name__ == '__main__':
    # super-poly
    p = Polynomial(np.array([1], dtype=np.float64))
    for sol in range(1, 4):
        p *= Polynomial(np.array([1, -sol], dtype=np.float64))
    print(p)
    x = np.linspace(-0.1, 2.001, 100)
    y = p(x)
    plt.plot(x, y)
    plt.show()
