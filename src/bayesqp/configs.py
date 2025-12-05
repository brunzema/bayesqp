import dataclasses
from enum import Enum
from dataclasses import dataclass, field
from typing import Union, Literal, Optional, Any

import torch
from torch import Tensor

from gpytorch.constraints import Interval
from gpytorch.kernels import ScaleKernel


_EPS = 1e-8

@dataclass
class LineSearchConfig:
    min_alpha: float = 0.0
    max_alpha: float = 1.0
    method: Literal["posterior_sampling"] = "posterior_sampling"
    n_candidates: int = 100
    M: int = 3
    log_scale: bool = False


@dataclass
class GeneralConfig:
    # Stopping criteria
    tol: float = 1e-4

    # Initialization
    n_initial: int = 5

    # Constraints & Bounds
    clamp_to_bounds: bool = True

    # Subsampling and SOCP config
    generate_local_samples_after_step: bool = True
    K: Union[int, Literal["auto"]] = "auto"
    epsilon: float = 0.05
    local_sample_strategy: Literal["sobol", "sobol_sphere", "global"] = "sobol_sphere"
    delta_f: float = 0.2
    delta_c: float = 0.2

    # Sliding window for GP
    N_max: Optional[int] = None

    # Flags
    use_resets: bool = False
    use_best_from_line_search: bool = True


class RBFKernelConfig(str, Enum):
    SQRT_D = "sqrt_D"  # https://arxiv.org/pdf/2402.02746
    D_SCALED_PRIOR = "D_scaled_prior"  # https://arxiv.org/pdf/2402.02229
    TURBO = "TuRBO_constraints"  # https://arxiv.org/pdf/1910.01739
    CUSTOM = "custom"


@dataclass
class KernelConfig:
    """Configuration specific to the covariance module."""

    type: Union[RBFKernelConfig, str] = RBFKernelConfig.D_SCALED_PRIOR

    # priors & constraints (not used e.g. for "d_scaled_prior" type)
    lengthscale_hyperprior: Optional[Any] = None
    outputscale_hyperprior: Optional[Any] = None
    outputscale_constraint: Optional[Interval] = None

    # Advanced
    provided_kernel: Optional[ScaleKernel] = None  # For 'custom' type


@dataclass
class ModelConfig:
    """General GP Model Configuration."""

    # Nested Kernel Configuration
    kernel: KernelConfig = field(default_factory=KernelConfig)

    # Likelihood / noise
    noise_constraint: Interval = field(default_factory=lambda: Interval(1e-6, 1e-1))
    noise_hyperprior: Optional[Any] = None

    # Mean
    prior_mean: float = 0.0


@dataclass
class OptimState:
    best_value: float = float("inf")
    best_constraint_values: Tensor = None
    best_value_index: int = -1
    num_constraints: int = 0
    best_feasible: bool = False

    def __post_init__(self):
        self.best_constraint_values = torch.full((self.num_constraints,), -float("inf"))


def print_current_defaults():
    """
    Instantiates all configuration dataclasses with their default values
    and prints them to get an overview of the configuration.
    """    
    configs = {
        "General Settings (general_config)": GeneralConfig(),
        "Line Search Settings (line_search_config)": LineSearchConfig(),
        "Model Settings (model_config)": ModelConfig(),
    }

    def _format_value(val):
        """Helper to format complex types for display."""
        if val is None:
            return "None"
        if isinstance(val, Enum):
            return f"'{val.value}'"
        if isinstance(val, torch.Tensor):
            if val.numel() == 1:
                return f"{val.item():.4g}"
            return f"Tensor shape={val.shape}"
        if hasattr(val, 'lower_bound') and hasattr(val, 'upper_bound'):
            lb = val.lower_bound.item() if torch.is_tensor(val.lower_bound) else val.lower_bound
            ub = val.upper_bound.item() if torch.is_tensor(val.upper_bound) else val.upper_bound
            return f"Interval[{lb:.1e}, {ub:.1e}]"
        return str(val)

    def _print_recursive(obj, indent=0):
        """Recursively prints dataclass fields."""
        prefix = "  " * indent
        
        for f in dataclasses.fields(obj):
            val = getattr(obj, f.name)
            
            if dataclasses.is_dataclass(val):
                print(f"{prefix}{f.name}:")
                _print_recursive(val, indent + 1)
            else:
                formatted_val = _format_value(val)
                print(f"{prefix}{f.name}: {formatted_val}")

    print("="*60)
    print(" BayeSQP Default Configurations")
    print("="*60)

    for title, config_obj in configs.items():
        print(f"\n[{title}]")
        _print_recursive(config_obj, indent=1)
    
    print("\n" + "="*60)

# Example Usage:
if __name__ == "__main__":
    print_current_defaults()