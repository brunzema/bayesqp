# Examples for BayeSQP

In this folder, we provide some examples on how to use BayeSQP. The package is build in a way that it works with any single-objective problem as long as it follows the [test function API from BoTorch (v0.15.1)](https://botorch.readthedocs.io/en/stable/test_functions.html#module-botorch.test_functions.base). Specifically, the problem has to inherrit from either `botorch.test_functions.base.ConstrainedBaseTestProblem` for constrained problem, or from `botorch.test_functions.base.BaseTestProblem` for unconstrained problems.

In the `standard_*.ipynb`, we show how to use BayeSQP on already implemented problems from BoTorch. In the `costum_*.ipynb`, we demonstrate how set up a costum problem using the BoTorch API (and of course then how to solve it with BayeSQP).
