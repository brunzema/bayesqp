# src/bayesqp/__init__.py

"""
BayeSQP: Bayesian Optimization through Sequential Quadratic Programming
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bayesqp")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"


from .bayesqp import BayeSQP
from .models import RBFHessianGPModel
from .utils import seed_everything
from .objective_wrappers import (
    NumpyToConstrainedBoTorchProblem,
    TorchToConstrainedBoTorchProblem,
)
from .configs import print_current_defaults

__all__ = [
    "BayeSQP",
    "RBFHessianGPModel",
    "seed_everything",
    "NumpyToConstrainedBoTorchProblem",
    "TorchToConstrainedBoTorchProblem",
    "print_current_defaults",
]
