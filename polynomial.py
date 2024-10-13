# from __future__ import annotations -> was used for " type-1 | type-2 " hinting in init and div-eval
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings

warnings.filterwarnings('error')


def sign(x):
    """Helper function - identical to np.sign."""
    # returns -1 if x < 0 else (1 if x > 0 else 0)
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0


class Polynomial:
    """
    A numpy-based polynomial class.
    The Polynomial object is a representation of a polynomial, using a numpy array to store the coefficients.
    """

    def __init__(self, coeffs: np.ndarray, rank=None, np_dtype=np.float64, eval_by='mm'):
        if coeffs is None:
            coeffs = np.array([0], dtype=np_dtype)  # default polynomial = 0
        if rank is None:
            rank = len(coeffs) - 1
        assert len(coeffs) == rank + 1, "Improper object instantiation, rank must be equal to len(coeffs) - 1"
        assert coeffs.dtype == np_dtype, \
            "When using a Polynomial coefficients array of a non np.float64 data-type," \
            " pass the chosen type to the constructor as np_dtype."
        if isinstance(coeffs, list):
            coeffs = np.array(coeffs, dtype=np_dtype)
        self.coeffs = coeffs
        self.rank = rank
        self.poly_size = rank + 1
        self.eval_by = eval_by
        self.reverse_eval_map = (self.coeffs[::-1], self.coeffs)  # idx=0 -> false, idx=1 -> true

    def __call__(self, *args, **kwargs):
        eval_at = args[0]
        if self.eval_by == 'horner':
            return self.poly_eval_horner(eval_at)
        return self.poly_eval(eval_at)

    def __repr__(self):
        """Returns the polynomial as a pretty string"""
        st = ""
        t_rank = self.rank
        for c1, c2 in zip(self.coeffs[:-1], self.coeffs[1:]):
            if c1 != 0:
                st += (f"{f'{c1:g}' if c1 != 1 else ''}x^{t_rank}" + (" +" if c2 > 0 else " "))
            t_rank -= 1
        st += f"{self.coeffs[-1]:g}"
        return st

    def poly_eval(self, at, do_inverse=0):
        # this is almost a whole second faster than at * np.ones(self.poly_size, dtype=self.coeffs.dtype)
        x_pows = np.empty(self.poly_size)
        x_pows.fill(at)
        x_pows[0] = np.float64(1)
        # fastest method
        x_pows = np.multiply.accumulate(x_pows, dtype=self.coeffs.dtype)
        # this is slower (instead of last row), but still faster than horner for large polynomials:
        # x_pows = x_pows ** np.arange(self.poly_size)
        # Rotem: modified from np.dot as this version seems to perform slightly better
        return np.vdot(x_pows, self.reverse_eval_map[do_inverse])

    def poly_eval_horner(self, at, do_inverse=0):
        res = 0
        _coeffs = self.reverse_eval_map[do_inverse]
        for val in _coeffs:
            res = (res * at) + val  # p(x) = ((((an)x) + an-1)x + an-2)
        return res

    def poly_sign(self, x):
        if abs(x) < 1:
            return sign(self(x))
        else:
            y = sign(self.inverse_poly(1 / x))
            if x > 0:
                return y
            return -y if self.rank & 1 else y

    def inverse_poly(self, at):
        if self.eval_by == 'horner':
            return self.poly_eval_horner(at, do_inverse=1)
        return self.poly_eval(at, do_inverse=1)

    def get_derivative_as_object(self):
        return Polynomial((np.arange(len(self.coeffs) - 1, -1, -1) * self.coeffs)[:-1],
                          self.rank - 1, eval_by=self.eval_by)

    def division_eval(self, denominator, at):
        """Evaluate p(x) / q(x) when both ranks are similar (same or about the same)."""
        if abs(at) < 1:
            denom = denominator(at)
            if denom == 0:
                return None
            return self(at) / denom
        if self.rank == denominator.rank + 1:
            dnm = denominator.inverse_poly(1 / at)
            if dnm == 0:
                return None
            return (at * self.inverse_poly(1 / at)) / dnm
        raise NotImplementedError(f"Improper ranks: {self.rank=}, {denominator.rank=}")

    def cauchy_bound(self):
        """A very quick-and-dirty method to find the bound for the absolute values of the solutions of a polynomial.
        Other, better bounds do exist, but require more care"""
        return 1 + np.max(np.abs(self.coeffs[1:] / self.coeffs[0]))

    def fujiwara_bound(self):
        """A slightly better bound (sometimes)."""
        a_n = self.coeffs[0]
        ones = np.ones(self.rank, dtype=self.coeffs.dtype)
        pows = ones / np.arange(1, self.poly_size)
        b = np.abs(self.coeffs[1:] / a_n)
        b[-1] /= 2
        return 2 * np.max(b ** pows)

    def get_absolute_root_bound(self):
        """Returns the tighter bound out of Cauchy and Fujiwara"""
        assert len(self.coeffs) and self.coeffs[0] != 0, "Cannot calculate Cauchy and " \
                                                         "Fujiwara's bound in case a_n is zero"
        a, b = self.cauchy_bound(), self.fujiwara_bound()
        return min(a, b)

    def get_normalized_poly(self, normalize_by=None):
        """Return a normalized polynomial object, with all coefficients divided by the maximum absolute value thereof"""
        if normalize_by is None:
            return Polynomial(self.coeffs / abs(self.coeffs[0]), eval_by=self.eval_by)
        assert normalize_by > 1
        return Polynomial(self.coeffs / normalize_by, eval_by=self.eval_by)

    def eval_many(self, at, do_inverse=0):
        """at is a vector (array)"""
        x_pows = np.multiply(at.reshape(-1, 1), np.tile(np.ones(self.poly_size), (len(at), 1)))
        x_pows[:, 0] = 1
        x_pows = np.multiply.accumulate(x_pows, axis=1)
        return np.dot(x_pows, self.reverse_eval_map[do_inverse])

    def poly_sign_many(self, at):
        """Gets all signs for at -> at is a vector (array). Attempt to vectorize poly-sign -
        no measurable performance improvement (since this does not avoid the loop)"""
        signs = np.empty(len(at))  # prepare signs array
        indices_abs_lt1 = abs(at) < 1
        indices_lt0 = ((at < 0) & ~indices_abs_lt1)
        indices_gt0 = ((at > 0) & ~indices_abs_lt1)
        signs[indices_abs_lt1] = np.sign(self.eval_many(at[indices_abs_lt1]))
        signs[indices_lt0] = np.sign(self.eval_many(1 / at[indices_lt0], 1))
        if self.rank % 2:
            signs[indices_lt0] *= -1
        signs[indices_gt0] = np.sign(self.eval_many(1 / at[indices_gt0], 1))
        return signs

    def height(self):
        return max(np.abs(self.coeffs))

    def separate_roots(self):
        # does not always work (very rough estimate)
        return 1 / (self.height())

    def descartes_sign_rule_count_roots(self):
        # positive roots:
        signs = np.sign(self.coeffs)
        # c = 0
        # for q,r in zip(signs[:-1], signs[1:]):
        #     if q != r:
        #         c += 1
        #
        # print(f"Positive sign changes: {c}")
        sign_changes = signs != np.roll(signs, -1)
        sign_changes[-1] = False  # remove last dummy
        pos_roots = len(signs[sign_changes])
        # negative roots:
        odd_sub = True if self.rank % 2 else False  # true if rank is odd, meaning we negate from the jump
        if odd_sub:
            signs[::2] *= -1
        else:
            signs[1::2] *= -1
        sign_changes = signs != np.roll(signs, -1)
        sign_changes[-1] = False  # remove last dummy
        neg_roots = len(signs[sign_changes])
        # print(f"Negative sign changes: {neg_roots}")

        # c2 = 0
        # for q,r in zip(signs[:-1], signs[1:]):
        #     if odd_sub:
        #         lft, rgt = -q, r
        #     else:
        #         lft, rgt = q, -r
        #
        #     if lft != rgt:
        #         c2 += 1
        # print(f"Negative sign changes: {c2}")
        return pos_roots, neg_roots

    """
    Both of the following will overflow!
    """
    def get_nth_derivative_native(self, k, normalize_at=1e+10):
        # Anx^n+An-1x^n-1+...+A1x+A0 -> N * An * (x^n-1) +
        # x^3 + x^2 + x + 1 -> 3x^2 + 2x+1 -> 6x + 2 + 1
        from math import factorial
        if k > self.rank:
            return [0]
        deriv = []
        for i, c in enumerate(self.coeffs):
            n = self.rank - i  # rank
            if n < k:
                break
            elm = (factorial(n) / factorial(n - k)) * c  # factorials times coefficient
            deriv.append(elm)
        return Polynomial(np.array(deriv, dtype=np.float64))
        # no normalization yet
        # pass

    def get_nth_derivative(self, n, normalize_at=1e+10):
        prep = np.arange(0, self.poly_size)
        prep[0] = 1
        if (t := math.factorial(n)) > normalize_at:
            prep /= t
        return np.multiply.accumulate(prep)[n:][::-1] * self.coeffs[n:]


if __name__ == '__main__':
    p = Polynomial(np.array([1, 1, 1, 1], dtype=np.float64))  # x^3 + x^2 + x + 1
    # print(p)
    # print(p.eval_many(np.array([1,2,3,4])))
    # print(p.poly_sign_many(np.array([-1, 1,2,3,4])))
    print(p.get_nth_derivative(2))
