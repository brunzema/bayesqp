import os
import random
from dataclasses import dataclass, fields, is_dataclass
from typing import List, Union

import numpy as np
import torch
from botorch.utils.transforms import normalize, unnormalize
from torch import Tensor


class BudgetExhaustedException(Exception):
    pass



def safe_cholesky(A, max_jitter=1e-1):
    """Compute Cholesky decomposition with jitter fallback and eigenvalue-based recovery.

    Args:
        A: Positive semi-definite matrix to decompose
        max_jitter: Maximum jitter to add before switching to eigenvalue decomposition

    Returns:
        Lower triangular Cholesky factor L such that A approx. L @ L.T
    """
    jitter = 1e-8
    n = A.shape[0]
    eye = torch.eye(n, dtype=A.dtype, device=A.device)

    # Try standard Cholesky with increasing jitter
    for _ in range(10):
        try:
            return torch.linalg.cholesky(A + eye * jitter)
        except RuntimeError:
            jitter *= 10
            if jitter > max_jitter:
                break

    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(A + eye * max_jitter)

        # Clamp negative eigenvalues to small positive value
        eigenvalues = eigenvalues.clamp(min=1e-10)

        # Compute L = Q @ diag(sqrt(λ))
        L = eigenvectors @ torch.diag(torch.sqrt(eigenvalues))

        return L

    except RuntimeError:
        import warnings

        warnings.warn(
            "Cholesky decomposition failed even with eigenvalue decomposition. "
            "Returning scaled identity matrix.",
            RuntimeWarning,
        )
        return eye * torch.sqrt(A.diag().mean())


def seed_everything(seed=0):
    """
    Sets the random seed for various libraries to ensure reproducibility.
    Copy paste: https://github.com/andresfp14/example/blob/main/modules/utils/seeds.py

    Args:
    - seed (int): Seed value. Default is 0.

    Note:
    This function sets seeds for the Python standard library, NumPy, and PyTorch.
    Additionally, it sets the environment variable for PL_GLOBAL_SEED.
    """

    # Set seed for the Python standard library's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # If CUDA is available, set the seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set environment variable for PyTorch Lightning's global seed
    os.environ["PL_GLOBAL_SEED"] = str(seed)


def numpy_wrapper(func):
    def wrapper(x):
        if isinstance(x, np.ndarray):
            return func(torch.from_numpy(x)).numpy()
        else:
            return func(x).squeeze().numpy()

    return wrapper


def convert_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().squeeze().numpy()
    if isinstance(x, list):
        # Convert each element in the list to numpy if possible
        return np.array(
            [convert_to_numpy(item) for item in x]
        )  # Keep object dtype to handle mixed types safely
    if isinstance(x, dict):
        return {key: convert_to_numpy(value) for key, value in x.items()}
    if is_dataclass(x):
        # Convert dataclass fields and reconstruct the dataclass
        converted_fields = {
            field.name: convert_to_numpy(getattr(x, field.name)) for field in fields(x)
        }
        return type(x)(**converted_fields)
    return x


def transform_x(x: Tensor | list, bounds: Tensor) -> Tensor | list:
    """
    Transform the input tensor using normalization.

    Args:
        x (Tensor): Input tensor.
    Returns:
        Tensor: Normalized tensor.
    """

    if isinstance(x, list):
        out = [normalize(xi, bounds) for xi in x]
        return out
    elif isinstance(x, Tensor):
        return normalize(x, bounds)
    else:
        raise ValueError(
            f"Unsupported type {type(x)}. Expected Tensor or list of Tensors."
        )


def retransform_x(x: Tensor | list, bounds: Tensor) -> Tensor | list:
    """
    Inverse transform the input tensor using unnormalization.

    Args:
        x (Tensor): Input tensor.
    Returns:
        Tensor: Unnormalized tensor.
    """
    if isinstance(x, list):
        out = [unnormalize(xi, bounds) for xi in x]
        return out
    elif isinstance(x, Tensor):
        return unnormalize(x, bounds)
    else:
        raise ValueError(
            f"Unsupported type {type(x)}. Expected Tensor or list of Tensors."
        )


@dataclass
class OptimizationInfo:
    """Information about the optimization process.

    Attributes:
        alphas: Step sizes used during optimization.
        convergence: Whether the optimizer converged or hit a limit.
        restarts: Number of restarts performed.
        n_initial: Number of initial samples.
        directions: Search directions explored.
        normalized_directions: Normalized search directions.
        gp_inference_time: Time spent on GP inference.
        model_train_time: Time spent training models.
        subsampling_time: Time spent on subsampling.
        ts_time: Time spent on trust region steps.
        sqp_time: Time spent solving SQP subproblems.
        hessian_time: Time spent computing Hessians.
        total_time: Total optimization time.
    """

    convergence: bool = None
    restarts: int = None
    n_initial: int = None
    total_time: float = None
    directions: Union[List, np.ndarray, Tensor] = None
    normalized_directions: Union[List, np.ndarray, Tensor] = None


@dataclass
class OptimizationResult:
    """Result of an optimization run using BayeSQP.

    Attributes:
        x: The best solution found.
        fun: The best objective value found.
        is_feasible: Whether the best solution satisfies all constraints.
        nfev: Number of function evaluations.
        history: History of all evaluated points.
        step_history: History of optimization steps.
        feasible_history: History of feasibility indicators for each point.
        fun_history: History of objective values.
        constraint_values: Values of constraints at evaluated points.
        info: Additional information about the optimization process.
    """

    x: Union[np.ndarray, Tensor, List]
    fun: float
    is_feasible: bool
    x_best_index: int
    nfev: int
    x_history: Union[List, np.ndarray, Tensor]
    step_history: Union[List, np.ndarray, Tensor]
    feasible_history: Union[List, np.ndarray]
    fun_history: Union[List, np.ndarray]
    constraint_values: Union[List, np.ndarray, Tensor]
    info: OptimizationInfo

    def __str__(self) -> str:
        """Pretty print the optimization result."""
        # Convert x to numpy for nice printing
        x_array = self._to_numpy(self.x)

        lines = [
            "=" * 70,
            "BayeSQP Optimization Result",
            "=" * 70,
            "",
            "Optimization Status:",
            f"  Feasible:        {'✓ Yes' if self.is_feasible else '✗ No'}",
            f"  Function Evals:  {self.nfev}",
            "",
            "Best Solution:",
            f"  Objective Value: {self.fun:.6e}",
            f"  Solution (x):    {self._format_array(x_array)}",
            f"  Best solution obtained at eval #{self.x_best_index}",
            "",
            f"Total Time: {self.info.total_time:.4f} seconds",
            "=" * 70,
        ]
        return "\n".join(lines)

    @staticmethod
    def _to_numpy(arr: Union[np.ndarray, Tensor, List]) -> np.ndarray:
        """Convert array-like to numpy array."""
        if isinstance(arr, Tensor):
            return arr.detach().cpu().numpy()
        elif isinstance(arr, list):
            return np.array(arr)
        return np.asarray(arr)

    @staticmethod
    def _format_array(arr: np.ndarray, max_items: int = 5) -> str:
        """Format array for nice printing."""
        if arr.size <= max_items:
            return np.array2string(
                arr, precision=4, suppress_small=True, separator=", "
            )
        else:
            # Show first and last elements for long arrays
            first_part = arr[:2]
            last_part = arr[-2:]
            first_str = np.array2string(
                first_part, precision=4, suppress_small=True, separator=", "
            )[1:-1]
            last_str = np.array2string(
                last_part, precision=4, suppress_small=True, separator=", "
            )[1:-1]
            return f"[{first_str}, ..., {last_str}]  (dim={arr.size})"
