import math
import time
import warnings

import numpy as np
from polynomial import Polynomial
import sys
from random import randint as random_randint

sys.setrecursionlimit(2000)

from time import perf_counter

dbg_time_bounds = 0
dbg_time_bisection = 0
dbg_time_div_eval = 0
dbg_bisection_iters = 0
dbg_newton_iters = 0
dbg_newton_calls = 0
dbg_newton_successes = 0
DEBUG_SHOW_STATS = True
DEBUG_SET_RANK = 996  # used for printing debug stats
VAL_TO_NORMALIZE = 1e+10  # divide polynomial by first coefficient if its magnitude is greater than this value


def root_finder_newton_raphson(polynomial: Polynomial, derivative: Polynomial, x, epsilon, sol_range):
    global dbg_time_div_eval, dbg_newton_iters, dbg_newton_successes, dbg_newton_calls
    dbg_newton_calls += 1
    iterations = int(np.sqrt(np.log10(1 / epsilon) + 1)) + 1  # assuming quadratic convergence
    for i in range(iterations):
        dbg_newton_iters += 1
        t = perf_counter()
        newton_step = polynomial.division_eval(derivative, x)
        dbg_time_div_eval += perf_counter() - t
        if newton_step is None or abs(newton_step) > sol_range:
            return None
        x -= newton_step
        if abs(newton_step) < epsilon:
            dbg_newton_successes += 1
            return x
    return None


def root_finder_bisection(polynomial: Polynomial, a, b, epsilon, a_sign, b_sign):
    global dbg_time_bisection, dbg_bisection_iters
    while (b - a) > epsilon:
        dbg_bisection_iters += 1
        mid = (b + a) / 2
        t = perf_counter()
        mid_sign = polynomial.poly_sign(mid)
        dbg_time_bisection += perf_counter() - t
        if a_sign * mid_sign > 0:
            a = mid
            a_sign = mid_sign
        else:
            b = mid
    return (a + b) / 2


def check_bound(polynomial: Polynomial, x, step, max_distance):
    x_sign = polynomial.poly_sign(x)
    if x_sign == np.sign(polynomial.poly_sign(max_distance * step)):
        return None
    return max_distance * step


def append_bounds_to_roots(derivative_roots, polynomial):
    poly_bound = polynomial.get_absolute_root_bound()
    left_bound = check_bound(polynomial, derivative_roots[0], -1, poly_bound)
    right_bound = check_bound(polynomial, derivative_roots[-1], 1, poly_bound)
    if left_bound is not None:
        derivative_roots = [left_bound] + derivative_roots
    if right_bound is not None:
        derivative_roots.append(right_bound)
    return derivative_roots


def polynomial_roots_finder_modified(polynomial: Polynomial, epsilon):
    """The difference her is just that I use normalization only
     once the large coefficient grows enough, not every time"""
    global dbg_time_bounds, dbg_time_bisection
    # base case - rank = 1
    if polynomial.rank == 1:  # ax+b -> -b/a
        return [-polynomial.coeffs[1] / polynomial.coeffs[0]]
    result_roots = []
    # grab unnormalized derivative
    unnormalized_derivative = polynomial.get_derivative_as_object()
    # normalize derivative in preparation for recursive call
    # if first coefficient reached a large enough value
    derivative_normalized_maybe = unnormalized_derivative.get_normalized_poly() \
        if unnormalized_derivative.coeffs[0] > VAL_TO_NORMALIZE else unnormalized_derivative
    # recursive call
    derivative_roots = polynomial_roots_finder_modified(
        derivative_normalized_maybe, epsilon)

    # if there are no zeros, pick a guess
    if not derivative_roots:
        derivative_roots = [0]
    # find tight bounds for the possible solutions (if exist)
    derivative_roots = append_bounds_to_roots(derivative_roots, polynomial)

    for i in range(len(derivative_roots) - 1):
        a, b = derivative_roots[i], derivative_roots[i + 1]
        a_sign = polynomial.poly_sign(a)
        b_sign = polynomial.poly_sign(b)
        # if np.sign(a_value) != np.sign(b_value):

        if a_sign != b_sign:
            # try Newton-Raphson for a few iterations
            t = perf_counter()
            root = root_finder_newton_raphson(polynomial, unnormalized_derivative, (a + b) / 2, epsilon, (b - a) / 2)
            dbg_time_bounds += perf_counter() - t
            # if N-R failed, use bisection to find root
            if root is None or not (a <= root <= b):
                root = root_finder_bisection(polynomial, a, b, epsilon, a_sign, b_sign)

            # add result to our list (we avoid duplicated via the above condition
            result_roots.append(root)

    # print a variety of debug info (specific for our test)
    if DEBUG_SHOW_STATS and polynomial.rank == DEBUG_SET_RANK:
        print(f"Time spend in newton: {dbg_time_bounds}")
        print(f"Time spend evaluating inside bisection : {dbg_time_bisection}")
        print(f"Time spent in newton's div-eval {dbg_time_div_eval}")
        print(f"Bisection ran for {dbg_bisection_iters} iterations in total!")
        print(f"Newton ran for {dbg_newton_iters} iterations in total!")
        print(f"Newton success rate: {100 * (dbg_newton_successes / dbg_newton_calls) : .4}%")
    return result_roots
