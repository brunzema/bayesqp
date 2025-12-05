import itertools
import math
import time
from typing import Dict, List, Union, TypeVar, Type

import gpytorch
import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.generation.sampling import (
    ConstrainedMaxPosteriorSampling,
    MaxPosteriorSampling,
)
from botorch.models.model import ModelList
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions.base import BaseTestProblem, ConstrainedBaseTestProblem
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import norm
from torch import Tensor
from torch.quasirandom import SobolEngine

from .models import NegateOutputModel, RBFHessianGPModel
from .subproblems import solve_bayesqp_unconstrained, solve_bayesqp_with_fallback
from .utils import (
    OptimizationInfo,
    OptimizationResult,
    BudgetExhaustedException,
    convert_to_numpy,
    retransform_x,
    transform_x,
)
from .configs import (
    GeneralConfig,
    LineSearchConfig,
    ModelConfig,
    OptimState
)


T = TypeVar("T")
_EPS = 1e-8
torch.set_default_dtype(torch.float64)


class BayeSQP:
    def __init__(
        self,
        objective_function: BaseTestProblem | ConstrainedBaseTestProblem,
        verbose_level: int = -1,
        general_config: Union[GeneralConfig, dict] = None,
        line_search_config: Union[LineSearchConfig, dict] = None,
        model_config: Union[ModelConfig, dict] = None,
    ):
        self.objective_function = objective_function
        self.dim = objective_function.dim
        self.verbose_level = verbose_level

        # load configs
        self.config = self._build_config(GeneralConfig, general_config)
        self.ls_config = self._build_config(LineSearchConfig, line_search_config)
        self.model_config = model_config
        
        # initialize State
        num_cons = getattr(objective_function, "num_constraints", 0)
        self.state = OptimState(num_constraints=num_cons)
        self.constrained_opt = num_cons > 0

        if self.config.K == "auto":
            self.n_subsamples = self.dim + 1
        else:
            self.n_subsamples = int(self.config.K)

        # result lists
        self.H_k = None
        self.history = []
        self.values = []
        self.constraint_values = []
        self.feasible = []
        self.fun_history = []
        self.step_history = []

        self.counter_f = 0
        self.converged_or_limit = None

    def _build_config(self, config_cls, input_data):
        """Helper to create config object from dict or existing object."""
        if input_data is None:
            return config_cls()
        if isinstance(input_data, dict):
            return config_cls(**input_data)
        return input_data

    def reset(self):
        self.history = []
        self.values = []
        self.constraint_values = []
        self.feasible = []
        self.fun_history = []
        self.counter_f = 0

    def _retransform_x(self, x: Tensor | list) -> Tensor | list:
        return retransform_x(x, self.objective_function.bounds)

    def _transform_x(self, x: Tensor | list) -> Tensor | list:
        return transform_x(x, self.objective_function.bounds)

    def _clamp_to_bounds(self, x: Tensor) -> Tensor:
        if self.config.clamp_to_bounds:
            return torch.clamp(
                x,
                min=torch.zeros_like(self.objective_function.bounds[0]),
                max=torch.ones_like(self.objective_function.bounds[1]),
            )
        else:
            return x

    def _evaluate_objective(self, x: Tensor) -> Tensor:
        if self.counter_f >= self.max_evals:
            self.converged_or_limit = True
            raise BudgetExhaustedException("Function evaluation budget exhausted")

        self.counter_f += 1
        x = self._clamp_to_bounds(x)
        x_original = unnormalize(x, self.objective_function.bounds)
        return self.objective_function(x_original).detach()

    def _evaluate_constraint(self, x: Tensor) -> Tensor:
        x = self._clamp_to_bounds(x)
        x_original = unnormalize(x, self.objective_function.bounds)
        return self.objective_function._evaluate_slack_true(x_original).detach()

    def _print(self, string: str, msg_type: str = "info"):
        if msg_type == "debug" and self.verbose_level >= 2:
            print(f"DEBUG:\t {string}")
        if msg_type == "model_info" and self.verbose_level >= 2:
            print(f"MODEL INFO:\t {string}")
        elif msg_type == "info" and self.verbose_level >= 0:
            print(f"INFO:\t {string}")
        elif msg_type == "warning" and self.verbose_level >= 1:
            print(f"WARNING:\t {string}")

    def _update_lists(self, x, y, c, feasible):
        self.history.append(x.clone().squeeze())
        self.values.append(y.item())
        self.constraint_values.append(c.squeeze())
        self.feasible.append(feasible)

    def _get_best_index_for_batch(self, Y: Tensor, C: Tensor | None) -> int:
        if self.constrained_opt:
            is_feas = (C >= 0).all(dim=-1)
            if is_feas.any():
                score = Y.clone()
                score[~is_feas] = float("inf")
                return score.argmin()
            return C.clamp(max=0).sum(dim=-1).argmax()
        else:
            return Y.argmin()

    def _update_state(self, X_next, Y_next, C_next, offset=0):
        # Pick the best point from the batch
        best_ind = self._get_best_index_for_batch(Y=Y_next, C=C_next)
        y_next = Y_next[best_ind]
        if not self.constrained_opt:
            C_next = torch.zeros_like(Y_next)

        c_next = C_next[best_ind]
        if X_next.shape[0] > 1:
            for i in range(X_next.shape[0]):
                self._update_lists(
                    x=X_next[i],
                    y=Y_next[i],
                    c=C_next[i],
                    feasible=(C_next[i] >= 0).all(),
                )
        else:
            self._update_lists(
                x=X_next,
                y=Y_next,
                c=C_next,
                feasible=(C_next >= 0).all(),
            )

        self._print(f"Best feasible value: {self.state.best_value}", msg_type="info")

        # Check if the best candidate is feasible
        is_feasible = (c_next >= 0).all()

        if is_feasible:
            # Only update if it's a better feasible solution
            if (
                not self.state.best_feasible  # No feasible solution before
                or y_next < self.state.best_value  # Better objective value
            ):
                self.state.best_value = y_next.item()
                self.state.best_feasible = True
                self.state.best_value_index = offset + best_ind
                self.state.best_constraint_values = c_next.clone()

        else:
            # No feasible candidate found; update if constraints are better
            total_violation_next = abs(c_next.clamp(max=0).sum(dim=-1))
            total_violation_best = abs(
                self.state.best_constraint_values.clamp(max=0).sum(dim=-1)
            )

            # Update best infeasible solution only if it's better
            if total_violation_next < total_violation_best:
                self.state.best_constraint_values = c_next.clone()

                # Only update best_value_index if no feasible solution exists yet
                if not self.state.best_feasible:
                    self.state.best_value_index = offset + best_ind

    def _generate_local_samples(
        self,
        x_k: Tensor,
        n: int,
        option: str = "sobol_sphere",
    ) -> Tensor:
        new_y = None
        constraints = None

        if option == "sobol":
            sobol_engine = SobolEngine(dimension=self.dim, scramble=True)
            sobol_samples = sobol_engine.draw(n)

            # Scale and shift Sobol samples to be within the small box around x_k
            train_x = x_k + (
                2 * self.config.epsilon * sobol_samples - self.config.epsilon
            )
        elif option == "global":
            # Generate Sobol sequence samples in [0,1]^dim
            sobol_engine = SobolEngine(dimension=self.dim, scramble=True)
            sobol_samples = sobol_engine.draw(n)
            train_x = sobol_samples
        elif option == "sobol_sphere":
            sobol_engine = SobolEngine(
                dimension=self.dim + 1, scramble=True
            )  # NOTE: we ned +1 for radius
            sobol_samples = sobol_engine.draw(n)

            # Convert first dim values to standard normal (Box-Muller or inverse CDF)
            normals = torch.erfinv(2 * sobol_samples[:, :-1] - 1) * np.sqrt(2)
            directions = normals / normals.norm(dim=1, keepdim=True)

            # Last dimension is for radius sampling
            r = sobol_samples[:, -1].pow(1.0 / self.dim) * self.config.epsilon

            # Sacle directions by radius and shift to x_k
            train_x = x_k + directions * r.unsqueeze(1)
        else:
            raise ValueError(f"Invalid option: {option}")

        if new_y is None:
            train_y = torch.tensor(
                [self._evaluate_objective(x) for x in train_x]
            ).unsqueeze(-1)

            if self.constrained_opt:
                constraints = torch.stack(
                    [self._evaluate_constraint(x) for x in train_x]
                )
            else:
                constraints = None

        return train_x, train_y, constraints

    def minimize(
        self, *args, max_evals: int = 20, return_as_numpy: bool = True, **kwargs
    ) -> Dict[str, Union[torch.Tensor, np.ndarray, float, int, List]]:
        self.max_evals = max_evals
        self.converged_or_limit = False
        res = self._minimize(*args, **kwargs)

        if return_as_numpy:
            res = convert_to_numpy(res)

        return res

    def _minimize(
        self,
        x0: Tensor,
    ) -> Dict[str, Tensor]:
        total_time = time.time()
        model_train_time = 0.0
        subsampling_time = 0.0
        ts_time = 0.0
        gp_inference_time = 0.0
        sqp_time = 0.0
        hessian_time = 0.0

        train_x = self._transform_x(x0)
        x_k = train_x.clone()
        constraints, next_constraints = None, None

        train_y = torch.tensor([self._evaluate_objective(train_x)]).unsqueeze(-1)

        if self.constrained_opt:
            constraints = torch.stack([self._evaluate_constraint(x) for x in train_x])
            self.step_is_feasible = (constraints >= 0).all()

        if self.config.n_initial > 0:
            new_x, new_y, next_constraints = self._generate_local_samples(
                x_k=x_k,
                n=self.config.n_initial,
                option=self.config.local_sample_strategy,
            )
            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, new_y])

            if self.constrained_opt:
                constraints = torch.cat([constraints, next_constraints])

        self._update_state(train_x, train_y, constraints, offset=0)

        p_ks = []
        restarts = [False] * len(train_x)
        self.step_history.append(x_k)
        step = 0
        lambda_k = None
        self.just_settled = False

        try:
            while True:
                t0 = time.time()
                self._update_models(train_x, train_y, constraints)
                model_train_time += time.time() - t0

                if lambda_k is None:
                    t0 = time.time()
                    lambda_next = (
                        np.zeros(self.objective_function.num_constraints)
                        if self.constrained_opt
                        else None
                    )
                    self._compute_hessian(
                        x_next=x_k,
                        lambda_next=lambda_next,
                    )
                    hessian_time += time.time() - t0
                else:
                    t0 = time.time()
                    lambda_next = lambda_k if self.constrained_opt else None
                    self._compute_hessian(
                        x_next=x_k,
                        lambda_next=lambda_next,
                    )
                    hessian_time += time.time() - t0

                # print gradient uncertainty (trace) for constraints and objective
                if self.verbose_level >= 2:
                    f_var = self.obj_model.posterior_derivative(x_k)[1][0]
                    self._print(
                        f"Objective grad uncertainty: {f_var.trace()}",
                        msg_type="debug",
                    )

                    if self.constrained_opt:
                        for i, c in enumerate(self.constraint_models):
                            c_var = c.posterior_derivative(x_k)[1][0]
                            self._print(
                                f"Constraint {i} grad uncertainty: {c_var.trace()}",
                                msg_type="debug",
                            )

                if self.constrained_opt:
                    res, info = solve_bayesqp_with_fallback(
                        bayesqp_object=self,
                        x_k=x_k,
                        delta_c=self.config.delta_c,
                        delta_f=self.config.delta_f,
                    )
                else:
                    res, info = solve_bayesqp_unconstrained(
                        bayesqp_object=self,
                        x_k=x_k,
                        delta_f=self.config.delta_f,
                    )

                sqp_time += info.get("sqp_time", 0.0)
                gp_inference_time += info.get("gp_inference_time", 0.0)

                lambda_k = res["lambda_c"]
                if res["status"] == "optimal":
                    p_k = res["p"]

                    scale_pk = False
                    if scale_pk:
                        scaling_factor = 1.0 / math.sqrt(self.dim)
                        p_k = p_k * scaling_factor
                else:
                    self._print(
                        "QP subproblem not optimal. Using random direction.",
                        msg_type="warning",
                    )
                    p_k = np.random.randn(self.dim) ** (1 / self.dim)

                p_ks.append(torch.from_numpy(p_k))

                self._print(f"--- Step {self.counter_f + 1} ---", msg_type="info")
                self._print(f"Norm: {np.linalg.norm(p_k)}", msg_type="debug")

                step_norm = np.linalg.norm(p_k)
                if self.config.use_resets and step_norm < self.config.tol:
                    self._print(
                        "-------------------------------------------------------",
                        msg_type="info",
                    )
                    self._print(
                        "Restarting optimization due to small step size.",
                        msg_type="info",
                    )
                    self._print(
                        "-------------------------------------------------------",
                        msg_type="info",
                    )
                    x_k = torch.rand_like(x_k)
                    new_x, new_y, next_constraints = self._generate_local_samples(
                        x_k=x_k,
                        n=self.dim + 1,
                        option=self.config.local_sample_strategy,
                    )
                    train_x = torch.cat([train_x, new_x])
                    train_y = torch.cat([train_y, new_y])

                    if self.constrained_opt:
                        constraints = torch.cat([constraints, next_constraints])

                    self._update_state(
                        new_x, new_y, next_constraints, offset=self.counter_f - 1
                    )

                    for i in range(len(new_x)):
                        restarts.append(True)

                    # make sure we dont double reset
                    self.just_settled = True
                elif step_norm < self.config.tol and self.state.best_feasible:
                    self.converged_or_limit = True
                else:
                    # Line search through constrained Thompson sampling
                    t0 = time.time()
                    sobol_engine = SobolEngine(dimension=1, scramble=True)
                    min_alpha = self.ls_config.min_alpha
                    max_alpha = self.ls_config.max_alpha

                    # Determine the maximum alpha that keeps points within [0,1]
                    # BSUB should handle this as well, but we do it here to be safe
                    max_safe_alpha = float("inf")
                    for d in range(self.dim):
                        # For each dimension, calculate the maximum alpha
                        if p_k[d] > 0:  # If moving in positive direction
                            max_dim_alpha = (1.0 - x_k[0, d]) / p_k[d]
                            max_safe_alpha = min(max_safe_alpha, max_dim_alpha)
                        elif p_k[d] < 0:  # If moving in negative direction
                            max_dim_alpha = -x_k[0, d] / p_k[d]
                            max_safe_alpha = min(max_safe_alpha, max_dim_alpha)

                    # Ensure max_safe_alpha is at least min_alpha and at most max_alpha
                    max_safe_alpha = min(max(max_safe_alpha, min_alpha), max_alpha)
                    pk_tensor = torch.tensor(p_k).reshape(1, -1)
                    for i in range(self.ls_config.M):
                        alpha_samples = sobol_engine.draw(self.ls_config.n_candidates)
                        alpha_values = (
                            min_alpha + (max_safe_alpha - min_alpha) * alpha_samples
                        )

                        # Map each alpha to a point on the line x_k + alpha * p_k
                        X_cand = torch.zeros(self.ls_config.n_candidates, self.dim)
                        for i, alpha in enumerate(alpha_values):
                            X_cand[i] = x_k + alpha.item() * pk_tensor

                        # Verify all points are within bounds
                        X_cand = torch.clamp(X_cand, 0.0, 1.0)

                        if self.constrained_opt:
                            negated_constraints = [
                                NegateOutputModel(c_model)
                                for c_model in self.constraint_models
                            ]
                            thompson_sampling = ConstrainedMaxPosteriorSampling(
                                model=NegateOutputModel(self.obj_model),
                                constraint_model=ModelList(*negated_constraints),
                                replacement=False,
                            )
                        else:
                            # If unconstrained we use standard TS
                            thompson_sampling = MaxPosteriorSampling(
                                model=NegateOutputModel(self.obj_model),
                                replacement=False,
                            )

                        with torch.no_grad():
                            x_next = thompson_sampling(X_cand, num_samples=1)

                        next_y = torch.tensor(
                            [self._evaluate_objective(x_next)]
                        ).unsqueeze(-1)

                        if self.constrained_opt:
                            next_constraints = torch.stack(
                                [self._evaluate_constraint(x) for x in x_next]
                            )

                        line_search_string = (
                            f"BO line search: Next x: {x_next}, Next y: {next_y}"
                        )
                        if self.constrained_opt:
                            violations = next_constraints < 0.0
                            sum_violations = next_constraints[violations].sum()
                            line_search_string += f", Next constraints: {next_constraints} (Sum: {sum_violations})"

                        self._print(line_search_string, msg_type="debug")

                        train_x = torch.cat([train_x, x_next])
                        train_y = torch.cat([train_y, next_y])
                        if self.constrained_opt:
                            constraints = torch.cat([constraints, next_constraints])

                        restarts.append(False)

                        # Update the state with the new point
                        self._update_state(
                            x_next, next_y, next_constraints, offset=self.counter_f - 1
                        )
                        self._update_models(train_x, train_y, constraints)

                    if self.config.use_best_from_line_search:
                        C = (
                            constraints[-self.ls_config.M :, :]
                            if self.constrained_opt
                            else None
                        )

                        # choose best candidate for next iteration
                        best_index_of_batch = self._get_best_index_for_batch(
                            Y=train_y[-self.ls_config.M :, :],
                            C=C,
                        )
                        idx = self.ls_config.M - best_index_of_batch
                        x_next = train_x[-idx, :].unsqueeze(0)
                        if self.constrained_opt:
                            next_constraints = constraints[-idx, :].unsqueeze(0)
                    else:
                        self._print(
                            f"Best feasible point: {self.state.best_value_index}",
                            msg_type="debug",
                        )
                        x_next = train_x[self.state.best_value_index, :].reshape(1, -1)
                        if self.constrained_opt:
                            next_constraints = constraints[
                                self.state.best_value_index, :
                            ].unsqueeze(0)

                    self._print(f"Next iterate: {x_next.numpy()}", msg_type="debug")

                    # 3. Update x_k for the next iteration
                    x_k = x_next
                    ts_time += time.time() - t0

                # Update step history
                self.step_history.append(x_k)

                # Convergence check
                if self.config.use_resets and not self.just_settled:
                    reset_conditions_fullfilled = self._check_reset_conditions()
                    if reset_conditions_fullfilled:
                        self._print(
                            "-------------------------------------------------------",
                            msg_type="info",
                        )
                        self._print(
                            "Restarting optimization due to close repeated steps.",
                            msg_type="info",
                        )
                        self._print(
                            "-------------------------------------------------------",
                            msg_type="info",
                        )
                        x_k = torch.rand_like(x_k)
                        new_x, new_y, next_constraints = self._generate_local_samples(
                            x_k=x_k,
                            n=self.n_subsamples,
                            option=self.subsampling_option,
                        )
                        train_x = torch.cat([train_x, new_x])
                        train_y = torch.cat([train_y, new_y])
                        if self.constrained_opt:
                            constraints = torch.cat([constraints, next_constraints])

                        self._update_state(
                            X_next=new_x,
                            Y_next=new_y,
                            C_next=next_constraints,
                            offset=self.counter_f - 1,
                        )

                        for i in range(len(new_x)):
                            restarts.append(True)

                # Check if we have converged or reached the limit
                if self.converged_or_limit:
                    self._print("Converged or limit reached.", msg_type="info")
                    break

                # Generate local samples around the new x_k
                if self.config.generate_local_samples_after_step:
                    self._print(
                        f"Generating {self.n_subsamples} local samples",
                        msg_type="debug",
                    )
                    t0 = time.time()
                    self._update_models(train_x, train_y, constraints)
                    model_train_time += time.time() - t0

                    # Generate local samples around the new x_k
                    t0 = time.time()
                    x_new, new_y, next_constraints = self._generate_local_samples(
                        x_k=x_k,
                        n=self.n_subsamples,
                        option=self.config.local_sample_strategy,
                    )
                    subsampling_time += time.time() - t0

                    train_x = torch.cat([train_x, x_new])
                    train_y = torch.cat([train_y, new_y])

                    if self.constrained_opt:
                        constraints = torch.cat([constraints, next_constraints])

                    self._update_state(
                        x_new,
                        new_y,
                        next_constraints,
                        offset=self.counter_f - len(x_new),
                    )

                # next QP iteration, can potentially reset again
                self.just_settled = False
                step += 1

        except BudgetExhaustedException:
            self._print(
                "Budget exhausted, returning current best result",
                msg_type="debug",
            )

        res = self._construct_result_dataclass(pk_s=p_ks, total_time=total_time)
        return res

    def _check_reset_conditions(self):
        n_past_points = math.ceil(max([4.0, float(self.dim)]))
        last_points = self.step_history[-n_past_points:]

        # Flatten to [dim] tensors for simplicity
        last_points = [p.squeeze(0) for p in last_points]

        # Check all pairwise distances
        for a, b in itertools.combinations(last_points, 2):
            if torch.norm(a - b) > self.tol:
                return False  # at least one pair is too far apart

        return True  # all pairwise distances are within epsilon

    def _get_constraint_probs(self, x_k):
        _, _, cs, cs_var = self.evaluate_point(x_k)
        constraint_probs = [norm.cdf(c / np.sqrt(v)) for c, v in zip(cs, cs_var)]
        return constraint_probs

    def _get_fitted_gp_model(self, train_x, train_y, type_model="constraint"):
        if self.config.N_max and len(train_x) > self.config.N_max:
            train_x = train_x[-self.config.N_max :, :]
            train_y = train_y[-self.config.N_max :, :]

        model = RBFHessianGPModel(
            train_X=train_x,
            train_Y=train_y,
            outcome_transform=Standardize(m=1),
            model_config=self.model_config
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        model.likelihood.noise = 0.0001

        try:
            with gpytorch.settings.max_cholesky_size(50):
                fit_gpytorch_mll(mll)
        except ModelFittingError:
            self._print(
                "Model fitting failed. Use default hyperparameters.",
                msg_type="info",
            )

        if self.verbose_level >= 2:
            lengthscales = (
                model.covar_module.base_kernel[0].lengthscale[0].clone().detach()
            )
            for i, lengthscale in enumerate(lengthscales):
                l_i = lengthscale.item()
                self._print(
                    f"Type:{type_model} Parameter name: Lengthscale {i} = {l_i:0.3f}",
                    msg_type="model_info",
                )
            self._print(
                f"Parameter name: Noise {model.likelihood.noise.item():0.3f}",
                msg_type="model_info",
            )

        model.eval()
        return model

    @property
    def surrogate_lengthscales(self):
        lengthscale_list = []

        if self.obj_model is not None:
            lengthscales = (
                self.obj_model.covar_module.base_kernel[0]
                .lengthscale[0]
                .clone()
                .detach()
            )
            lengthscale_list.append(lengthscales)

        if self.constrained_opt:
            for i, c in enumerate(self.constraint_models):
                lengthscales = (
                    c.covar_module.base_kernel[0].lengthscale[0].clone().detach()
                )
                lengthscale_list.append(lengthscales)

        # Stack to (num_models, dim), then aggregate to (dim,)
        stacked = torch.stack(lengthscale_list, dim=0)  # (num_models, dim)
        return stacked.mean(dim=0)

    def _update_models(self, train_x, train_y, constraints):
        self.obj_model = self._get_fitted_gp_model(
            train_x=train_x,
            train_y=train_y,
            type_model="objective",
        )

        if self.constrained_opt:
            self.constraint_models = [
                self._get_fitted_gp_model(train_x, constraints[:, i].unsqueeze(-1))
                for i in range(self.objective_function.num_constraints)
            ]

    def evaluate_point(self, x):
        """
        Evaluate objective and constraint models at a given point.

        Args:
            x: The point to evaluate.

        Returns:
            tuple: (obj_mean, obj_std, constraint_means, constraint_stds)
        """
        # Evaluate objective model
        obj_pred = self.obj_model.posterior(x)
        obj_mean = obj_pred.mean.item()
        obj_std = obj_pred.variance.sqrt().item()

        # Evaluate constraint models
        if self.constrained_opt:
            constraint_means = []
            constraint_stds = []
            for c in self.constraint_models:
                pred = c.posterior(x)
                mean_pred = pred.mean.detach().numpy().flatten()
                constraint_means.append(mean_pred)
                constraint_stds.append(pred.variance.sqrt().detach().numpy().item())

        else:
            constraint_means = None
            constraint_stds = None

        return obj_mean, obj_std, constraint_means, constraint_stds

    def evaluate_gradient(self, x):
        # Evaluate objective model
        with torch.no_grad():
            obj_mean, obj_var = self.obj_model.posterior_derivative(x)
            obj_mean = obj_mean[0].numpy()
            obj_var = obj_var[0].numpy()

            # Evaluate constraint models
            if self.constrained_opt:
                constraint_preds = [
                    c.posterior_derivative(x) for c in self.constraint_models
                ]
                constraint_means = [pred[0][0].numpy() for pred in constraint_preds]
                constraint_vars = [pred[1][0].numpy() for pred in constraint_preds]
            else:
                constraint_means = None
                constraint_vars = None

        return obj_mean, obj_var, constraint_means, constraint_vars

    def _compute_hessian(self, x_next, lambda_next):
        # Ensure x is a torch tensor
        if not isinstance(x_next, torch.Tensor):
            x_next = torch.tensor(x_next, dtype=torch.float64)

        # Get hessian of objective
        with torch.no_grad():
            hessian_f = self.obj_model.posterior_hessian_mean(x_next)
            hessian_f = hessian_f.numpy().squeeze()

        hessian_L = hessian_f.copy()

        if lambda_next is not None:
            for i, constraint_model in enumerate(self.constraint_models):
                if lambda_next[i] > _EPS:
                    with torch.no_grad():
                        hessian_c = constraint_model.posterior_hessian_mean(x_next)
                        hessian_c = hessian_c.numpy().squeeze()

                    lambda_i = lambda_next[i]
                    hessian_L -= lambda_i * hessian_c

        # Add small regularization to diagonal before checking eigenvalues
        reg_lambda = _EPS
        hessian_L_reg = hessian_L + reg_lambda * np.eye(self.dim)

        eigvals = np.linalg.eigvals(hessian_L_reg)
        if np.any(eigvals < 0):
            hessian_L_reg = eigen_positive_definite(hessian_L_reg)

        # Ensure symmetry
        self.H_k = hessian_L_reg

    def _retransform_direction(self, d: Tensor | list) -> Tensor | list:
        """Transform direction vectors from normalized space to original space."""
        bounds = self.objective_function.bounds
        bounds_range = bounds[1] - bounds[0]

        if isinstance(d, list):
            return [direction * bounds_range for direction in d]
        else:
            return d * bounds_range

    def _construct_result_dataclass(
        self, pk_s: List[np.ndarray] = [], total_time: float = 0.0
    ) -> Dict[str, Union[torch.Tensor, float, int, List]]:
        info = OptimizationInfo(
            convergence=self.converged_or_limit,
            n_initial=self.config.n_initial,
            normalized_directions=pk_s,
            directions=self._retransform_direction(pk_s),
            total_time=time.time() - total_time,
        )

        res = OptimizationResult(
            x=self._retransform_x(self.history[self.state.best_value_index]),
            fun=self.state.best_value,
            is_feasible=self.state.best_feasible,
            x_best_index=self.state.best_value_index,
            nfev=self.counter_f,
            x_history=self._retransform_x(self.history),
            step_history=self._retransform_x(self.step_history),
            feasible_history=self.feasible,
            fun_history=self.values,
            constraint_values=self.constraint_values,
            info=info,
        )
        return res


def eigen_positive_definite(H, min_eig=1e-3):
    """
    Ensures a matrix is positive definite by modifying its eigenvalues.

    Args:
        H: Input Hessian matrix
        min_eig: Minimum eigenvalue threshold

    Returns:
        Modified positive definite matrix
    """
    # Compute eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(H)

    # Replace negative eigenvalues with min_eig
    modified_eigvals = np.maximum(eigvals, min_eig)

    # Reconstruct the matrix
    modified_H = eigvecs.dot(np.diag(modified_eigvals)).dot(eigvecs.T)
    modified_H = (modified_H + modified_H.T) / 2  # Ensure symmetry
    return modified_H
