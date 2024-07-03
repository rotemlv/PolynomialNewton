import numpy as np


def newtons_method_original(x0, f, f_prime, tolerance, epsilon, max_iterations):
    """Newton's method

    Args:
      x0:              The initial guess
      f:               The function whose root we are trying to find
      f_prime:         The derivative of the function
      tolerance:       Stop when iterations change by less than this
      epsilon:         Do not divide by a number smaller than this
      max_iterations:  The maximum number of iterations to compute
    """
    for _ in range(max_iterations):
        y = f(x0)
        yprime = f_prime(x0)

        if abs(yprime) < epsilon:       # Give up if the denominator is too small
            break

        x1 = x0 - y / yprime            # Do Newton's computation

        if abs(x1 - x0) <= tolerance:   # Stop when the result is within the desired tolerance
            return x1                   # x1 is a solution within tolerance and maximum number of iterations

        x0 = x1                         # Update x0 to start the process again

    return None                         # Newton's method did not converge

def newtons_method(x0, f, f_prime, tolerance, epsilon, max_iterations):
    """Newton's method (Wikipedia implementation - modified by me)
    Args:
      x0:              The initial guess
      f:               The function whose root we are trying to find
      f_prime:         The derivative of the function
      tolerance:       Stop when iterations change by less than this
      epsilon:         Do not divide by a number smaller than this
      max_iterations:  The maximum number of iterations to compute
    """
    # print(f"Called newtons_method with: {x0=}, {f=}, {f_prime=}, types= {type(x0)}, {type(f)}, {type(f_prime)}")
    assert abs(f.coeffs[0] * f.rank - f_prime.coeffs[0]) < epsilon
    original_guess = x0
    iter_values = []
    for _ in range(max_iterations):
        try:
            yprime = f_prime(x0)
        except RuntimeWarning:
            print(f"Failed in newton!")
            exit(-1)
            # return None
        if abs(yprime) < epsilon:  # Give up if the denominator is too small
            print("Failed in derivative value!")
            return None
        try:
            x1 = x0 - f.division_eval(f_prime, x0)  # Do Newton's computation
        except RuntimeWarning as w:
            if "divide by zero" in str(w):
                print("Failed in derivative value calculation (got 0 in denominator)")
                return None

        if abs(x1 - x0) < tolerance:  # Stop when the result is within the desired tolerance
            return x1  # x1 is a solution within tolerance and maximum number of iterations
        # print(f"{x0=} -> {x1=}")

        x0 = x1  # Update x0 to start the process again
        iter_values.append(x0)
    print(f"Failed in max-iterations for {original_guess=}, reached: {x1 if max_iterations>0 else x0}")
    # print(f"Iteration values: {iter_values[-30:]}")
    # if np.float64(10.191941417363024) == original_guess:
    #     print(iter_values)
    return None  # Newton's method did not converge
