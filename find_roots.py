from poly import *
from newton_raphson import *
from math import log as math_log
VAL_TO_NORMALIZE = 10_000
EPSILON = 1e-8
MAX_ITERATIONS = 100# * int( math_log(1 / EPSILON) / math_log(2) )


def find_all_roots(polynom: Polynomial, eps: float):
    print(f"Entered call with polynomial: {polynom if polynom.rank < 5 else polynom.rank}")
    if polynom.rank == 0:
        return Polynomial(np.array([], dtype=polynom.coeffs.dtype))
    if polynom.rank == 1:
        # p(x) = ax + b -> 0 = ax+b -> x = -b/a
        return np.array([- polynom.coeffs[1] / polynom.coeffs[0]], dtype=polynom.coeffs.dtype)
    # normalize:
    if polynom.coeffs[0] > VAL_TO_NORMALIZE:
        # REPLACE POLY WITH COPY!
        # (this broke everything because derivative is normalized but original function isn't),
        # got huge numbers in div
        print("Normalization performed!")
        polynom = polynom.get_normalized_poly()
    der_obj = polynom.get_derivative_as_object()
    zero_derivative_points = find_all_roots(der_obj, eps)
    print(f"Left recursive call for polynom: {polynom if polynom.rank < 5 else polynom.rank}")
    # print(f"Found zero derivative points at: {zero_derivative_points}")
    # find bounds:
    upper = polynom.get_absolute_root_bound()
    lower = - upper
    # pad the lower to left and append upper to right:
    possible_roots = np.append(np.pad(zero_derivative_points, (1, 0), mode='constant', constant_values=lower), upper)
    # use bisection to get midpoints:
    bisection_points = (possible_roots[1:] + possible_roots[:-1]) / 2
    print(f"Bisection points (points to test): {np.round(bisection_points, 3)}, len={len(bisection_points)}")
    roots = np.array([], dtype=np.float64)
    for i, point in enumerate(bisection_points):
        potential_root = newtons_method(point, polynom, der_obj,
                                        eps, eps, MAX_ITERATIONS)
        # solution_not_close =
        if (potential_root is not None) and\
                (not len(roots) or not (np.all(np.isclose(potential_root, roots, rtol=2 * eps)))):
            roots = np.append(roots, potential_root)
        elif potential_root is None:
            # do grid search?

            pass

    return roots


# tmp1, tmp2 = Polynomial(np.array([1, 2, 1])), Polynomial(np.array([1, -2]))
# p = Polynomial(np.array([1,2,1]))
# p = tmp1 * tmp2
# super-poly
p = Polynomial(np.array([1], dtype=np.float64))
for sol in range(1, 16):
    p *= Polynomial(np.array([1, -sol], dtype=np.float64))
print(p)
print(find_all_roots(p, np.float64(EPSILON)))