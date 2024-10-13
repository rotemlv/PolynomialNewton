import numpy as np
from polynomial import Polynomial
import sys

sys.setrecursionlimit(2000)  # safety margin for using recursion - avoid limit errors
VAL_TO_NORMALIZE = 1e+10  # divide polynomial by first coefficient if its magnitude is greater than this value


def root_finder_newton_raphson(polynomial: Polynomial, derivative: Polynomial, x, epsilon, sol_range, iterations):
    for i in range(iterations):
        newton_step = polynomial.division_eval(derivative, x)
        if newton_step is None or abs(newton_step) > sol_range:
            return None
        if abs(newton_step) < epsilon:
            return x - newton_step
        x -= newton_step
    return None


def root_finder_bisection(polynomial: Polynomial, a, b, epsilon, a_sign):
    while (b - a) / 2 > epsilon:
        mid = (b + a) / 2
        mid_sign = polynomial.poly_sign(mid)
        if a_sign == mid_sign:
            a = mid
            a_sign = mid_sign
        else:
            b = mid
    return (b + a) / 2


def check_bound(polynomial: Polynomial, x, step, bound):
    """For a given polynomial, edge-root (from left or right), side and bound - > returns edge-bound if it exists."""
    if polynomial.poly_sign(x) == np.sign(polynomial.poly_sign(bound * step)):
        return None
    return bound * step


def append_bounds_to_roots(derivative_roots, polynomial):
    poly_bound = polynomial.get_absolute_root_bound()
    left_bound = check_bound(polynomial, derivative_roots[0], -1, poly_bound)
    right_bound = check_bound(polynomial, derivative_roots[-1], 1, poly_bound)
    if left_bound is not None:
        derivative_roots = [left_bound] + derivative_roots
    if right_bound is not None:
        derivative_roots.append(right_bound)
    return derivative_roots


def polynomial_roots_finder(polynomial: Polynomial, epsilon, newton_iterations=2):
    """The difference her is just that I use normalization only
     once the large coefficient grows enough, not every time"""
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
    derivative_roots = polynomial_roots_finder(
        derivative_normalized_maybe, epsilon, newton_iterations)

    # if there are no zeros, pick a guess
    if not derivative_roots:
        derivative_roots = [0]
    # find tight bounds for the possible solutions (if exist)
    derivative_roots = append_bounds_to_roots(derivative_roots, polynomial)
    # pre-calculating the signs provides a slight speed boost (using np.sign)
    poly_signs = np.sign([polynomial.poly_sign(x) for x in derivative_roots])
    for i in range(len(derivative_roots) - 1):
        a, b = derivative_roots[i], derivative_roots[i + 1]
        a_sign, b_sign = poly_signs[i], poly_signs[i + 1]
        # a_sign = polynomial.poly_sign(a)
        # b_sign = polynomial.poly_sign(b)
        if a_sign != b_sign:
            # try Newton's method for a few iterations
            root = root_finder_newton_raphson(polynomial, unnormalized_derivative, (a + b) / 2, epsilon, (b - a) / 2,
                                              newton_iterations)
            # if N-R failed, use bisection to find root
            if root is None:
                root = root_finder_bisection(polynomial, a, b, epsilon, a_sign)
            # add result to our list
            result_roots.append(root)
    return result_roots


def create_all_derivatives(polynomial):
    derivs = [polynomial]
    last = polynomial
    for r in range(1, polynomial.rank):
        curr = last.get_derivative_as_object()
        curr = curr if curr.coeffs[0] < VAL_TO_NORMALIZE else curr.get_normalized_poly(VAL_TO_NORMALIZE)
        derivs.append(curr)
        last = curr
    return derivs


def polynomial_roots_finder_iterative(polynomial: Polynomial, epsilon, newton_iterations=2):
    """The difference her is just that I use normalization only
     once the large coefficient grows enough, not every time"""
    # start from case 1 - derivative of rank poly.rank - 1!
    result_roots = []
    get_derivative = create_all_derivatives(polynomial)
    linear_case = get_derivative[polynomial.rank - 1]
    prev_deriv = linear_case
    derivative_roots = [-linear_case.coeffs[1] / linear_case.coeffs[0]]
    for r in range(2, polynomial.rank + 1):
        # get next derivative
        poly_deriv = get_derivative[polynomial.rank - r]
        # get bounds for roots of "next deriv"
        if not derivative_roots:
            # if there are no zeros, pick a guess
            derivative_roots = [0]
        derivative_roots = append_bounds_to_roots(derivative_roots, poly_deriv)
        result_roots = []
        # iterate over possible roots (below loop)
        # pre-calculating the signs provides a slight speed boost (using np.sign)
        poly_signs = [poly_deriv.poly_sign(x) for x in derivative_roots]
        for i in range(len(derivative_roots) - 1):
            a, b = derivative_roots[i], derivative_roots[i + 1]
            a_sign, b_sign = poly_signs[i], poly_signs[i + 1]
            if a_sign != b_sign:
                # try Newton's method for a few iterations
                root = root_finder_newton_raphson(poly_deriv, prev_deriv, (a + b) / 2, epsilon,
                                                  (b - a) / 2,
                                                  newton_iterations)
                # if N-R failed, use bisection to find root
                if root is None:
                    root = root_finder_bisection(poly_deriv, a, b, epsilon, a_sign)
                # add result to our list
                result_roots.append(root)
        # prep for next iteration
        # assign solutions to next deriv
        derivative_roots = result_roots
        prev_deriv = poly_deriv

    return result_roots


def maybe_deterministic_newton(poly: Polynomial, epsilon, space_ratio=2, step=1, min_root_dist=0.001):  # try mrd=0.001
    """This root-finding method can only work when a polynomial P(x) has roots that are distributed
    in an almost uniform way, or a polynomial with root number N << rank(P).
    Modify the "space_ratio" multiplier to improve versatility, though it will greatlydecrease performance.
    Performance delta is: T(n, c) = c * T(n, 1).
    Cannot guarantee convergence as specific polynomials constructed s.t.
    all roots are inside the linspace's epsilon.
    (linspace(a, b, total_count) creates an array [a,a+(b-a)/total_count, ...] -
    if the distance between two elements in it contains more than one root, we will not find it.

    Improvement -
    Use a proper root separation bound (minimal distance between roots).
    This concept allows this method to be deterministic ONLY IF there are no memory limitations (too large sol. space).
    """
    bound = poly.get_absolute_root_bound()
    a_s, b_s = poly.poly_sign(-bound), poly.poly_sign(bound)
    # tighten bounds as  much as possible:
    tmp_a = -bound
    tmp_b = bound
    while tmp_a < tmp_b and poly.poly_sign(tmp_a) == a_s:
        tmp_a += step
    a = tmp_a - step
    while tmp_a < tmp_b and poly.poly_sign(tmp_b) == b_s:
        tmp_b -= step
    b = tmp_b + step
    if min_root_dist is None:
        # obtain a possible solutions space
        space = np.linspace(a, b, 1 + (space_ratio * poly.rank))  # adding 1 since we need n sign changes
    else:
        # use min-dist to define space
        total_size = int(1 + (b - a) / min_root_dist)
        space = np.linspace(a, b, 1 + total_size)  # adding 1 since we need n sign changes so n+1 elements

    # calculate signs for it
    space_signs = poly.poly_sign_many(space)
    diff_signs = space_signs != np.roll(space_signs, -1)
    diff_signs[-1] = False
    # isolate locations of changed signs (as candidate roots)
    filtered_space = space[diff_signs]

    roots = []
    # this is our "tolerance" for a given solution (not to leave a specific range while newton is iterating)
    space_width = space[1] - space[0]
    derivative = poly.get_derivative_as_object()
    for candidate in filtered_space:
        # we now know that each candidate represents a spot around which the sign changes
        newton_root = root_finder_newton_raphson(poly, derivative, candidate,
                                                 epsilon, space_width,
                                                 20)  # 20 newton iterations cover pretty much any epsilon
        if newton_root is not None:
            roots.append(newton_root)
        else:
            roots.append(
                root_finder_bisection(poly,
                                      candidate - space_width,  # a
                                      candidate + space_width,  # b
                                      epsilon,
                                      poly.poly_sign(candidate - space_width)  # sign(p(a))
                                      )
            )

    return roots


def maybe_deterministic_newton_parallel(poly: Polynomial,
                                        epsilon, space_ratio=1,
                                        step=1, min_root_dist=1e-4,  # mrd set to 5e-5 for "realistic" use
                                        num_workers=4):  # try mrd=0.01
    """This root-finding method can only work when a polynomial P(x) has roots that are distributed
    in an almost uniform way, or a polynomial with root number N << rank(P).
    Modify the "space_ratio" multiplier to improve versatility, though it will greatly decrease performance.
    Performance delta is: T(n, c) = c * T(n, 1).
    (if min_root_dist is not None, performance decrease is inversely proportional to it instead)
    Cannot guarantee convergence as specific polynomials constructed s.t.
    all roots are inside the linspace's epsilon.
    (linspace(a, b, total_count) creates an array [a,a+(b-a)/total_count, ...] -
    if the distance between two elements in it contains more than one root, we will not find all of them.

    Improvements -
    A. Use a proper root separation bound (minimal distance between roots).
    This concept allows this method to be deterministic ONLY IF there are no memory limitations (too large sol. space),
    though an iterative approach would work.
    B. Define subgroups of bounds using derivative signs (this can be done recursively, though it will prob. be slow).
    """
    bound = poly.get_absolute_root_bound()
    a_s, b_s = poly.poly_sign(-bound), poly.poly_sign(bound)
    # tighten bounds as  much as possible:
    tmp_a = -bound
    tmp_b = bound
    while tmp_a < tmp_b and poly.poly_sign(tmp_a) == a_s:
        tmp_a += step
    a = tmp_a - step
    while tmp_a < tmp_b and poly.poly_sign(tmp_b) == b_s:
        tmp_b -= step
    b = tmp_b + step
    if min_root_dist is None:
        # obtain a possible solutions space
        space = np.linspace(a, b, (space_ratio * poly.poly_size))  # size not rank, for worst case (deg(p) sign changes)
        # reason for size is that we need at least one more point than total sign change instances,
        # in order to perform future calculation (we use the space rolled over itself to get sign changes)
    else:
        # use min-dist to define space
        total_size = int(1 + (b - a) / min_root_dist)
        space = np.linspace(a, b, 1 + total_size)  # adding 1 since we need n sign changes so n+1 elements

    # calculate signs for it
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as p:
        res = np.empty(len(space))
        split_spaces = np.array_split(space, num_workers)
        # print(split_spaces)
        idx = 0
        for (s, r) in zip(split_spaces, p.map(poly.poly_sign_many, split_spaces)):
            res[idx:idx+len(s)] = r
            idx += len(s)
        space_signs = res

    # isolate locations of changed signs (as candidate roots)
    # TODO: this is slower than serial calculation (too simple for multiproc?)
    # with ProcessPoolExecutor(max_workers=num_workers) as p:
    #     res = np.empty(len(space), dtype=np.bool_)
    #     split_space_signs = np.array_split(space_signs[:-1], num_workers)
    #     split_rolled_space_signs = np.array_split(space_signs[1:], num_workers)
    #
    #     # print(split_spaces)
    #     idx = 0
    #     for (s, r) in zip(split_space_signs, p.map(np.not_equal, split_space_signs, split_rolled_space_signs)):
    #         res[idx:idx+len(s)] = r
    #         idx += len(s)
    #     sign_diffs = res
    #     sign_diffs[-1] = False
    sign_diffs = space_signs != np.roll(space_signs, -1)  # a, b, c cmp b, c, a (we don't need last comparison!)
    sign_diffs[-1] = False # forcing last element to not be a sign change (edge case)
    filtered_space = space[sign_diffs]


    # print(f"Length of space: {len(space)}, width = {space[1]-space[0]}")
    roots = []
    # this is our "tolerance" for a given solution (not to leave a specific range while newton is iterating)
    space_width = space[1] - space[0]
    derivative = poly.get_derivative_as_object()
    for candidate in filtered_space:
        # we now know that each candidate represents a spot around which the sign changes
        newton_root = root_finder_newton_raphson(poly, derivative, candidate,
                                                 epsilon, space_width,
                                                 20)  # 20 newton iterations cover pretty much any epsilon
        if newton_root is not None:
            roots.append(newton_root)
        else:
            roots.append(
                root_finder_bisection(poly,
                                      candidate - space_width,  # a
                                      candidate + space_width,  # b
                                      epsilon,
                                      poly.poly_sign(candidate - space_width)  # sign(p(a))
                                      )
            )

    return roots
