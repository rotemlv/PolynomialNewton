from root_finder import *
import time

epsilon = 1.0e-12
SHOW_NUMPY_RESULTS = True
EVAL_METHOD = 'mm'
SHOW_SAME_ORDER = True


def main():
    with open('poly_coeff(997).txt') as f:
        polynomial = f.read().splitlines()
    polynomial_coefficients = np.array(polynomial, np.longdouble)
    if SHOW_NUMPY_RESULTS:
        start = time.time()
        # this works for numpy version < 2, otherwise remove the "ndarray" part
        np_roots = np.roots(np.ndarray.astype(polynomial_coefficients, float))
        end = time.time()
        print("\nnp.roots() (using companion matrix):\n")
        for r in np_roots[np.isreal(np_roots)]:
            print(f"({r.real})")

        print()
        print(f"np.roots() runtime is: {end - start} seconds\n")

    print(
        f"Performing root finding using {('mat-mul' if EVAL_METHOD == 'mm' else 'horner method')} as polynomial "
        f"evaluation method")

    newton_iterations = int(np.sqrt(np.log10(1 / epsilon) + 1))  # assuming perfect quadratic convergence

    start = time.time()
    results = polynomial_roots_finder(Polynomial(polynomial_coefficients, eval_by=EVAL_METHOD),
                                      epsilon,
                                      newton_iterations)
    end = time.time()

    print("Newton-Raphson with bisection fallback roots:\n")
    if SHOW_NUMPY_RESULTS and SHOW_SAME_ORDER:
        results = sorted(results, key=lambda x: abs(x), reverse=True)
    for r in results:
        print(f"({r})")
    print()
    print(f"The runtime is: {end - start} seconds\n")

    print("Now running newton-raphson non deterministic:")
    start = time.time()
    results = maybe_deterministic_newton(Polynomial(polynomial_coefficients, eval_by=EVAL_METHOD),
                                         epsilon,
                                         )
    end = time.time()

    print("Newton-Raphson NON-DET roots:\n")
    if SHOW_NUMPY_RESULTS and SHOW_SAME_ORDER:
        results = sorted(results, key=lambda x: abs(x), reverse=True)
    for r in results:
        print(f"({r})")
    print()
    print(f"The runtime is: {end - start} seconds")

    tmp = Polynomial(polynomial_coefficients, eval_by=EVAL_METHOD)
    print(f"\nBound root sep (WORK IN PROGRESS):"
          f" {tmp.separate_roots()}")

    print(f"Descartes' law of signs gave me pos,neg: {tmp.descartes_sign_rule_count_roots()}")


if __name__ == '__main__':
    main()
