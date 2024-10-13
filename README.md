## PolynomialNewton
This repository includes the implementation I presented in the course "Programming Languages" (Braude, final semester), but also the following:
- arbitrary precision polynomial class, can be used for better root approximations after finding a set of roots using a reasonable epsilon via the newton-raphson method
- multi-processing-based approach for newton-raphson solver: this uses matrix operations and ditches the recursion in favor of a grid-search approach, performs much faster on some cases, will run out of memory in others.

This repo also includes the example polynomial shown during the course presentation.
