from poly import *
from newton_raphson import *
from math import log as math_log
import sys

VAL_TO_NORMALIZE = 1e+30
EPSILON = 1e-7
MAX_ITERATIONS = 30  # * int( math_log(1 / EPSILON) / math_log(2) )
MAX_BISECTION_ITERATIONS = 100_000
sys.setrecursionlimit(2000)
DO_VERBOSE = True
# verbose bisection prints a lot so it's best used only for deliberate debugging cases
VERBOSE_BISECTION = False

def get_sign_diff(poly, point, st, en):
    """

    :param poly: function to eval
    :param point: point in interval
    :param st: start of interval
    :param en: end of interval
    :return: True if point*st < 0, False if point*en < 0 else None
    """
    curr = np.sign(poly(point))
    lft = np.sign(poly(st))
    rgt = np.sign(poly(en))
    if curr * lft < 0:
        return True
    elif curr * rgt < 0:
        return False
    return None


def in_range(newton_root, foo_range):
    st, en = foo_range
    return st <= newton_root <= en


def bisection_method(polynom: Polynomial, lft: np.number, rgt: np.number, eps: np.number, tol: np.number = None):

    # xxprint = print if DO_VERBOSE else (lambda x, sep=None, end=None: None)
    xxprint = print if VERBOSE_BISECTION else (lambda x, sep=None, end=None: None)
    if tol is None:
        tol = eps * eps
    xxprint(f"Entered bisection with {lft}, {rgt}")
    # assert that the range is appropriate for polynomial
    # meaning, f(st)*f(en) < 0
    if polynom(lft) * polynom(rgt) > 0:
        xxprint("Cannot run bisection - interval does not change signs")
        return None
    # find mid, f(mid) and f(edges)
    mid = (lft + rgt) / 2
    curr = polynom(mid)
    val_lft, val_rgt = polynom(lft), polynom(rgt)
    # edge cases - solution is at the edges:
    if abs(val_lft) < eps:
        return lft
    if abs(val_rgt) < eps:
        return rgt
    it = 0
    # TODO: remove mid condition
    while abs(lft - rgt) > tol and abs(curr) > eps and it < MAX_BISECTION_ITERATIONS:
        # increment iterations counter
        it += 1
        xxprint(f"{mid=}, {lft=}, {rgt=}")
        old_mid = mid
        curr_sign = 1 if curr > 0 else -1
        sign_left = 1 if val_lft > 0 else -1
        sign_right = 1 if val_rgt > 0 else -1
        is_increasing = (sign_left == -1) and (sign_right == 1)
        if is_increasing:
            # func is increasing and sign does changes, decrease interval from right
            xxprint("Function is increasing, ",end='')
            if curr_sign == 1:
                xxprint("midpoint is positive, setting right endpoint to mid")
                rgt = mid
                val_rgt = curr
            # oppositely, decrease from left (sign did not change yet)
            else:
                xxprint("midpoint is negative, setting left endpoint to mid")
                lft = mid
                val_lft = curr
        # func is decreasing - conditions behave oppositely
        elif (sign_left == 1) and (sign_right == -1):
            xxprint("Function is decreasing, ",end='')
            if curr_sign == 1:
                xxprint("midpoint is positive, setting left endpoint to mid")
                lft = mid
                val_lft = curr
            else:
                xxprint("midpoint is positive, setting right endpoint to mid")
                rgt = mid
                val_rgt = curr
        else:
            # we WERE in a changing interval but not anymore. I don't know how to handle this case.
            # This happens in regions of oscillations with bad precision
            return mid if polynom(mid + eps) * polynom(mid - eps) < 0 else None
        mid = (lft + rgt) / 2
        if old_mid == mid:
            return mid
        curr = polynom(mid)

    # mid leaves tolerance easily
    return mid if polynom(mid + eps) * polynom(mid - eps) < 0 else None

# p = Polynomial([1,-1])
# p = p * Polynomial([1, -2]) # (x-1)(x-2) -> der = x-2 + x-1 -> sol at 3/2
# g = 0
# en = p.get_absolute_root_bound()
# st = - en
# print(bisection_method(p, st, 3/2, 1e-5))
# exit()


def find_all_roots(polynom: Polynomial, eps: float):
    xxprint = print if DO_VERBOSE else (lambda x, sep=None, end=None: None)
    xxprint(f"Entered call at rank: {polynom.rank}")
    if polynom.rank == 0:
        return np.array([], dtype=polynom.coeffs.dtype)  # Polynomial(np.array([], dtype=polynom.coeffs.dtype))
    if polynom.rank == 1:
        # p(x) = ax + b -> 0 = ax+b -> x = -b/a
        return np.array([- polynom.coeffs[1] / polynom.coeffs[0]], dtype=polynom.coeffs.dtype)
    # normalize:
    if polynom.coeffs[0] > VAL_TO_NORMALIZE:
        xxprint("Normalization performed!")
        polynom = polynom.get_normalized_poly()
    der_obj = polynom.get_derivative_as_object()
    zero_derivative_points = find_all_roots(der_obj, eps)
    xxprint(f"Left recursive call for rank: {polynom.rank}")
    xxprint(f"zero derivative points :{zero_derivative_points}")
    # print(f"Found zero derivative points at: {zero_derivative_points}")
    # find bounds:
    upper = polynom.get_absolute_root_bound()
    lower = - upper
    # pad the lower to left and append upper to right:
    possible_roots = np.append(np.pad(zero_derivative_points, (1, 0), mode='constant', constant_values=lower), upper)
    xxprint(f"Points of interest :{possible_roots}")

    # use bisection to get midpoints:
    roots = []
    for lft, rgt in zip(possible_roots[:-1], possible_roots[1:]):
        mid = (lft + rgt) / 2
        # try newton:
        xxprint(f"Trying {MAX_ITERATIONS} Newton iterations!")
        newton_root = newtons_method(mid, polynom, der_obj, eps, eps, MAX_ITERATIONS, DO_VERBOSE)
        xxprint(f"Newton returned {newton_root}!")
        if newton_root and in_range(newton_root, (lft, rgt)) and all(abs(rt - newton_root) > 2 * eps for rt in roots):
            roots.append(newton_root)
        else:
            # do bisection
            bisection_root = bisection_method(polynom, lft, rgt, eps)
            xxprint(f"Bisection for {lft}, {rgt} returned: {bisection_root}")
            if bisection_root is None:
                x = np.linspace(lft, rgt, 10000)
                y = polynom(x)
                # y /= np.min(y)
                # plt.plot(x,y)
                # plt.show()
                with open("test.txt",'w') as f:
                    f.write(str(polynom)+'\n')
                    for _x, _y in zip(x,y):
                        f.write(f"{_x}: {_y}\n")
                xxprint(f"Wrote to file 'test.txt' the poly's values in the range (left, right). "
                        f"Check for oscillations.")
                exit()
            if bisection_root and all(
                    abs(rt - bisection_root) > 2 * eps for rt in roots):
                roots.append(bisection_root)

    return np.array(roots, dtype=polynom.coeffs.dtype)


# tmp1, tmp2 = Polynomial(np.array([1, 2, 1])), Polynomial(np.array([1, -2]))
# p = Polynomial(np.array([1,2,1]))
# p = tmp1 * tmp2
# super-poly
p = Polynomial(np.array([1], dtype=np.float64))
for sol in range(1, 1000):
    p *= Polynomial(np.array([1, -sol], dtype=np.float64))
print(p)
print(find_all_roots(p, np.float64(EPSILON)))
