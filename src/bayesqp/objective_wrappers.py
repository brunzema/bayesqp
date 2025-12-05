import torch
import numpy as np
from typing import Callable, List, Tuple, Optional
from torch import Tensor

from botorch.test_functions.base import ConstrainedBaseTestProblem


class NumpyToConstrainedBoTorchProblem(ConstrainedBaseTestProblem):
    """
    A wrapper class to convert generic NumPy objective and constraint functions 
    into a BoTorch ConstrainedBaseTestProblem.
    """

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        constraint_funcs: List[Callable[[np.ndarray], float]],
        bounds: List[Tuple[float, float]],
        noise_std: Optional[float | List[float]] = None,
        constraint_noise_std: Optional[float | List[float]] = None,
        negate: bool = False,
        check_grad_at_opt: bool = False
    ) -> None:
        """
        Args:
            objective_func: Function taking a 1D (d,) numpy array, returning a float.
            constraint_funcs: List of functions taking a 1D (d,) numpy array. 
                              Must return positive values for feasible entries (c(x) >= 0).
            bounds: List of (min, max) tuples for each dimension.
            noise_std: Standard deviation of observation noise for objective.
            constraint_noise_std: Standard deviation of observation noise for constraints.
            negate: If True, negates the objective (for maximization).
        """
        self.dim = len(bounds)
        self._bounds = bounds
        self.num_constraints = len(constraint_funcs)
        
        # store the numpy functions
        self._np_objective = objective_func
        self._np_constraints = constraint_funcs
        
        # assuming all parameters are continuous for this wrapper
        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []
        
        self._check_grad_at_opt = check_grad_at_opt
        self.constraint_noise_std = constraint_noise_std
        super().__init__(noise_std=noise_std, negate=negate)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        """
        Evaluates the objective function by converting Tensor -> Numpy,
        running the function, and converting Numpy -> Tensor.
        """
        # X shape: (batch_shape) x d
        # We flatten batch dimensions to loop easily, then reshape back.
        original_shape = X.shape[:-1]
        
        # Flatten to (N, d) for iteration
        X_flat = X.view(-1, self.dim)
        X_np = X_flat.detach().cpu().numpy()
        
        # Evaluate objective row by row
        values_np = np.array([self._np_objective(x) for x in X_np])
        
        # Convert back to tensor with same device/dtype as input
        values = torch.tensor(
            values_np, 
            dtype=X.dtype, 
            device=X.device
        )
        
        # Reshape to (batch_shape)
        return values.view(original_shape)

    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        """
        Evaluates constraints by converting Tensor -> Numpy.
        Returns tensor where > 0 indicates feasibility.
        """
        original_shape = X.shape[:-1]
        
        X_flat = X.view(-1, self.dim)
        X_np = X_flat.detach().cpu().numpy()
        
        # Evaluate all constraints for every x in the batch
        slack_rows = []
        for x in X_np:
            row_slacks = [c_func(x) for c_func in self._np_constraints]
            slack_rows.append(row_slacks)
            
        slack_np = np.array(slack_rows)
        
        # Convert back to tensor
        slacks = torch.tensor(
            slack_np, 
            dtype=X.dtype, 
            device=X.device
        )
        
        # Back to (batch_shape) x num_constraints
        return slacks.view(*original_shape, self.num_constraints)
    
    
class TorchToConstrainedBoTorchProblem(ConstrainedBaseTestProblem):
    """
    A wrapper class to convert generic PyTorch objective and constraint functions 
    into a BoTorch ConstrainedBaseTestProblem.
    """

    def __init__(
        self,
        objective_func: Callable[[Tensor], Tensor],
        constraint_funcs: List[Callable[[Tensor], Tensor]],
        bounds: List[Tuple[float, float]],
        noise_std: Optional[float | List[float]] = None,
        constraint_noise_std: Optional[float | List[float]] = None,
        negate: bool = False,
        vectorized: bool = False
    ) -> None:
        """
        Args:
            objective_func: Function taking a Tensor. Returns a scalar (or batch of scalars).
            constraint_funcs: List of functions taking a Tensor. Returns slack (val >= 0 is feasible).
            bounds: List of (min, max) tuples.
            vectorized: 
                If False (default): The wrapper assumes your functions only handle a single point 
                (shape `d`).
                If True: The wrapper passes the full batch `(..., d)` to your function. 
                Your function must handle dimensions correctly.
        """
        self.dim = len(bounds)
        self._bounds = bounds
        self.num_constraints = len(constraint_funcs)
        
        self._obj_func = objective_func
        self._cons_funcs = constraint_funcs
        self._vectorized = vectorized
        
        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []
        self.constraint_noise_std = constraint_noise_std

        super().__init__(noise_std=noise_std, negate=negate)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        if self._vectorized:
            return self._obj_func(X)
        
        # If not vectorized, loop over batch dimensions
        # X shape: (..., d) -> Flatten to (N, d)
        original_shape = X.shape[:-1]
        X_flat = X.view(-1, self.dim)
        
        # Calculate one by one
        values = torch.stack([self._obj_func(x) for x in X_flat])
        
        # Reshape to (...)
        return values.view(original_shape)

    def _evaluate_slack_true(self, X: Tensor) -> Tensor:
        if self._vectorized:
            # We stack them to get (batch_shape x num_constraints)
            slacks = [c(X) for c in self._cons_funcs]
            return torch.stack(slacks, dim=-1)

        # If not vectorized, loop over batch dimensions
        original_shape = X.shape[:-1]
        X_flat = X.view(-1, self.dim)
        
        batch_slacks = []
        for x in X_flat:
            point_slacks = torch.stack([c(x) for c in self._cons_funcs])
            batch_slacks.append(point_slacks)
            
        # Stack all points
        all_slacks = torch.stack(batch_slacks) # (N, num_constraints)
        
        # Reshape to (... x num_constraints)
        return all_slacks.view(*original_shape, self.num_constraints)