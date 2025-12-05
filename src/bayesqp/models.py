import math
from typing import Union, Optional, Any

import gpytorch
import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_lognormal_prior,
)
from gpytorch.constraints import Interval
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from torch import Tensor
from torch.distributions import MultivariateNormal

from .configs import ModelConfig, KernelConfig, RBFKernelConfig

torch.set_default_dtype(torch.float64)


def build_rbf_kernel(
    config: Union[KernelConfig, str],
    dims: int,
    lengthscale_hyperprior: Optional[Any] = None,
    outputscale_constraint: Optional[Interval] = None,
    outputscale_hyperprior: Optional[Any] = None,
    provided_kernel: Optional[ScaleKernel] = None,
) -> ScaleKernel:
    """
    Factory function to build the specific RBF kernel configuration.
    Returns a ScaleKernel wrapping an RBFKernel. Note that tis is for now necessary
    to support the hard-coded derivative and Hessian calculations.
    """
    # Ensure config is Enum
    if isinstance(config, str):
        try:
            config = KernelConfig(config)
        except ValueError:
            raise ValueError(f"Unknown kernel config: {config}. Options: {[e.value for e in KernelConfig]}")

    if config.type == RBFKernelConfig.SQRT_D:
        base_kernel = RBFKernel(
            ard_num_dims=dims,
            lengthscale_prior=lengthscale_hyperprior,
            lengthscale_constraint=Interval(0.001, 2 * dims),
        )
        base_kernel.lengthscale = math.sqrt(dims)
        return ScaleKernel(
            base_kernel,
            outputscale_constraint=outputscale_constraint,
            outputscale_prior=outputscale_hyperprior,
        )

    elif config.type == RBFKernelConfig.D_SCALED_PRIOR:
        d_scaled_rbf = get_covar_module_with_dim_scaled_prior(
            use_rbf_kernel=True,
            ard_num_dims=dims,
        )
        return ScaleKernel(
            d_scaled_rbf,
            outputscale_prior=outputscale_hyperprior,
            outputscale_constraint=outputscale_constraint
        )

    elif config.type == RBFKernelConfig.TURBO:
        return ScaleKernel(
            RBFKernel(
                ard_num_dims=dims,
                lengthscale_prior=lengthscale_hyperprior,
                lengthscale_constraint=Interval(0.001, 4),
            ),
            outputscale_constraint=outputscale_constraint,
            outputscale_prior=outputscale_hyperprior,
        )

    elif config.type == RBFKernelConfig.CUSTOM:
        if provided_kernel is None:
            raise ValueError("Configuration set to 'custom' but 'provided_kernel' is None.")
        
        if not isinstance(provided_kernel, ScaleKernel) or not isinstance(provided_kernel.base_kernel, RBFKernel):
             raise ValueError("Provided kernel must be ScaleKernel(RBFKernel) to support derivative/Hessian calculations.")
        
        return provided_kernel
    
    raise ValueError(f"Unreachable code for config: {config}")


class RBFGPModel(ExactGP, GPyTorchModel):
    _num_outputs = 1

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        model_config: ModelConfig = None,
        outcome_transform=None,
        input_transform=None,
    ):
        self.cfg = self._build_config(model_config)
        
        # setup likelihood
        likelihood = GaussianLikelihood(
            noise_constraint=self.cfg.noise_constraint,
            noise_prior=self.cfg.noise_hyperprior
        )
        
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=None)
        with torch.no_grad():
            transformed_X = self.transform_inputs(X=train_X, input_transform=input_transform)
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(Y=train_Y, Yvar=None, X=transformed_X)
        self._validate_tensor_args(X=transformed_X, Y=train_Y, Yvar=None)
        
        super().__init__(train_X, train_Y.squeeze(-1), likelihood)
        
        self.D = train_X.shape[1]
        if outcome_transform:
            self.outcome_transform = outcome_transform
        if input_transform:
            self.input_transform = input_transform

        self.mean_module = gpytorch.means.ConstantMean()
        if self.cfg.prior_mean is not None:
            self.mean_module.initialize(constant=self.cfg.prior_mean)
            self.mean_module.constant.requires_grad = False

        # setup kernel        
        self.covar_module = build_rbf_kernel(
            config=self.cfg.kernel.type,
            ard_num_dims=self.D,
            lengthscale_hyperprior=self.cfg.kernel.lengthscale_hyperprior,
            outputscale_constraint=self.cfg.kernel.outputscale_constraint,
            outputscale_hyperprior=self.cfg.kernel.outputscale_hyperprior,
            provided_kernel=self.cfg.kernel.provided_kernel
        )

        self.covar_module.outputscale = 1.0
        self.to(train_X)

    def _build_config(self, cfg_in):
        """Helper to ensure we have a ModelConfig object."""
        if cfg_in is None: 
            return ModelConfig()
        if isinstance(cfg_in, dict): 
            k_cfg = cfg_in.pop('kernel', {})
            m_cfg = ModelConfig(**cfg_in)
            if isinstance(k_cfg, dict):
                m_cfg.kernel = KernelConfig(**k_cfg)
            return m_cfg
        return cfg_in

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def get_mean_function(self):
        """Return the mean function of the GP as a callable function."""

        def posterior_mean_function(x):
            with torch.no_grad():
                return self.posterior(x).mean

        return posterior_mean_function


class RBFDerivativeGPModel(RBFGPModel):
    """Derivative of the RBFGPModel w.r.t. input x.
    This model is mainly based on the implementation used for
    GIBO (NeurIPS 2021) (https://github.com/sarmueller/gibo)."""

    def append_train_data(self, new_x, new_y):
        """Append training data with optional normalization."""
        return self.condition_on_observations(self.transform_inputs(new_x), new_y)

    def posterior_derivative(self, X):
        """Compute posterior derivative of GP at given test points x."""
        X = self.transform_inputs(X)
        if self.prediction_strategy is None:
            self.posterior(X)

        K_xX_dx = self._get_KxX_dx(X)  # shape: (batch, D, n_train)
        KXX_inv = self.get_KXX_inv()  # shape: (n_train, n_train)
        mean_d = K_xX_dx @ KXX_inv @ self.train_targets  # shape: (batch, D)
        K_xx_dx2 = self._get_Kxx_dx2()  # shape: (D, D)
        temp = torch.matmul(K_xX_dx, KXX_inv)  # shape: (batch, D, n_train)
        variance_d = (
            K_xx_dx2.unsqueeze(0) - torch.bmm(temp, K_xX_dx.transpose(1, 2))
        ).clamp_min(1e-9)  # shape: (batch, D, D)

        # Apply outcome transform
        if hasattr(self, "outcome_transform"):
            _, stdvs, stdvs_sq = self.outcome_transform._get_per_input_means_stdvs(
                X=X, include_stdvs_sq=True
            )
            mean_d = mean_d * stdvs.unsqueeze(-1)
            variance_d = variance_d * stdvs_sq.unsqueeze(-1).unsqueeze(-1)
            mean_d = mean_d.squeeze(-1)
            variance_d = variance_d.squeeze(-1)

        return mean_d[0], variance_d[0]

    def get_KXX_inv(self):
        """Compute inverse of K(X, X)."""
        L_inv_upper = self.prediction_strategy.covar_cache.detach()
        return L_inv_upper @ L_inv_upper.transpose(0, 1)

    def _get_KxX_dx(self, x):
        """Compute analytic derivative of K(x, X) w.r.t. x."""
        X = self.train_inputs[0]
        x = self.transform_inputs(x)
        K_xX = self.covar_module(x, X).evaluate()

        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        if lengthscale.dim() == 2:
            lengthscale = lengthscale.squeeze(0)

        diff = x.unsqueeze(1) - X.unsqueeze(0)  # (batch, n_train, D)
        scaled_diff = diff / (lengthscale.unsqueeze(0).unsqueeze(0) ** 2)

        # Apply kernel values
        K_expanded = K_xX.unsqueeze(-1)  # (batch, n_train, 1)
        return -scaled_diff.transpose(1, 2) * K_expanded.transpose(1, 2)

    def _get_Kxx_dx2(self):
        """Compute analytic second derivative of K(x, x) w.r.t. x."""
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        if lengthscale.dim() == 2:
            lengthscale = lengthscale.squeeze(0)
        sigma_f = self.covar_module.outputscale.detach()

        return torch.diag(1.0 / (lengthscale**2)) * sigma_f

    def get_mean_gradient_function(self):
        """Compute the mean of the derivative GP as a callable function."""

        def posterior_mean_gradient_function(x):
            with torch.no_grad():
                return self.posterior_derivative(x)[0]

        return posterior_mean_gradient_function

    def joint_covariance(self, x, ignore_crossterms=False):
        with torch.no_grad():
            return self._return_joint_covariance(x, ignore_crossterms)

    def _return_joint_covariance(self, x, ignore_crossterms=False):
        joint_cov_f = torch.zeros((self.D + 1, self.D + 1), device=x.device)

        # standard prediction
        pred = self.posterior(x)
        pred_covar = pred.mvn.covariance_matrix
        joint_cov_f[0, 0] = pred_covar.squeeze()

        # derivative prediction
        _, grad_covar = self.posterior_derivative(x)

        # Fill in the derivatives covariance (lower-right DxD block)
        joint_cov_f[1:, 1:] = grad_covar.squeeze()

        if not ignore_crossterms:
            # Compute cross-covariance between function value and derivatives
            X = self.train_inputs[0]
            K_xX = self.covar_module(x, X).evaluate()  # shape: (batch_size, n)
            K_xX_dx = self._get_KxX_dx(x)  # shape: (batch_size, D, n)

            # Compute cross-covariance
            KXX_inv = self.get_KXX_inv()  # shape: (n, n)
            temp = K_xX @ KXX_inv  # shape: (batch_size, n)
            cross_covar = -temp @ K_xX_dx.transpose(1, 2).squeeze()

            # Fill in cross-covariance terms
            joint_cov_f[0, 1:] = cross_covar
            joint_cov_f[1:, 0] = cross_covar

        # Apply outcome transform
        # TODO: Add maybe support for other transforms?
        if hasattr(self, "outcome_transform"):
            _, stdvs, stdvs_sq = self.outcome_transform._get_per_input_means_stdvs(
                X=x, include_stdvs_sq=True
            )

            # Apply transformation
            stdvs = stdvs.squeeze()
            stdvs_sq = stdvs_sq.squeeze()
            joint_cov_f[0, 0] = joint_cov_f[0, 0] * stdvs_sq
            joint_cov_f[1:, 1:] = joint_cov_f[1:, 1:] * stdvs_sq
            joint_cov_f[0, 1:] = joint_cov_f[0, 1:] * stdvs_sq
            joint_cov_f[1:, 0] = joint_cov_f[1:, 0] * stdvs_sq

        return joint_cov_f


class RBFHessianGPModel(RBFDerivativeGPModel):
    """Implementation of Hessian computation for GP.
    NOTE: Currently only consideres the mean Hessian."""

    def __init__(
        self,
        train_X,
        train_Y,
        outcome_transform=None,
        input_transform=None,
        **kwargs,
    ):
        super().__init__(
            train_X,
            train_Y,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            **kwargs,
        )

    def posterior_hessian_mean(self, X):
        """Compute posterior Hessian mean using vectorized operations."""
        X = self.transform_inputs(X)
        if self.prediction_strategy is None:
            self.posterior(X)

        # Get kernel parameters
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        if lengthscale.dim() == 2:
            lengthscale = lengthscale.squeeze(0)

        # Compute Hessian kernel efficiently
        X_train = self.train_inputs[0]
        K_xX = self.covar_module(X, X_train).evaluate()  # (batch, n_train)

        # Compute differences and scaled differences
        diff = X.unsqueeze(1) - X_train.unsqueeze(0)  # (batch, n_train, D)
        inv_ls2 = 1.0 / (lengthscale**2)  # (D,)

        # Vectorized Hessian kernel computation
        scaled_diff = diff * inv_ls2.unsqueeze(0).unsqueeze(0)  # (batch, n_train, D)

        # Outer product for cross-derivative terms
        outer_product = scaled_diff.unsqueeze(-1) * scaled_diff.unsqueeze(
            -2
        )  # (batch, n_train, D, D)

        # Diagonal terms
        eye = torch.eye(self.D, device=X.device)
        diagonal_term = -eye * inv_ls2.unsqueeze(-1)  # (D, D)

        # Combine terms
        K_expanded = K_xX.unsqueeze(-1).unsqueeze(-1)  # (batch, n_train, 1, 1)
        K_hessian = K_expanded * (
            outer_product + diagonal_term.unsqueeze(0).unsqueeze(0)
        )

        # Compute mean
        KXX_inv = self.get_KXX_inv()
        train_targets = self.train_targets
        mean_h = torch.einsum("bnij,nm,m->bij", K_hessian, KXX_inv, train_targets)

        # Apply outcome transformation if necessary
        if hasattr(self, "outcome_transform"):
            _, stdvs, _ = self.outcome_transform._get_per_input_means_stdvs(
                X=X, include_stdvs_sq=False
            )
            mean_h = mean_h * stdvs.unsqueeze(-1).unsqueeze(-1)

        return mean_h[0]

    def get_mean_hessian_function(self):
        """Return callable function for Hessian mean."""

        def posterior_mean_hessian_function(x):
            with torch.no_grad():
                result = self.posterior_hessian_mean(x)
                # Handle single vs batch input
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                if result.dim() == 3 and x.shape[0] == 1:
                    return result[0]
                return result

        return posterior_mean_hessian_function


class NegateOutputModel(Model):
    """A model wrapper that negates the outputs of a base model.
    BayeSQP is formulated to minimize and the current version of constrained
    Thompson sampling in BoTorch only supports maximization."""

    def __init__(self, base_model: Model):
        super().__init__()
        self.base_model = base_model

    def posterior(self, X: Tensor, **kwargs):
        base_posterior = self.base_model.posterior(X, **kwargs)
        cov = base_posterior.mvn.covariance_matrix
        jitter = 1e-6 * torch.eye(cov.size(-1), device=cov.device)
        stable_cov = cov + jitter
        inverted_mvn = MultivariateNormal(
            loc=-base_posterior.mvn.mean,
            covariance_matrix=stable_cov,
        )
        return type(base_posterior)(inverted_mvn)

    def condition_on_observations(self, X, Y, **kwargs):
        conditioned = self.base_model.condition_on_observations(X, -Y, **kwargs)
        return NegateOutputModel(conditioned)

    def subset_output(self, indices: list[int]):
        return NegateOutputModel(self.base_model.subset_output(indices))

    @property
    def num_outputs(self):
        return self.base_model.num_outputs
