import time
import traceback

import numpy as np
import scipy
import scipy.special
from cvxopt import matrix, solvers

"""To solve the uncertainty-aware subproblem from BayeSQP (BSUB), we leverage the
second-order quadratic cone solver from CVXOPT. For the documentation, see:
https://cvxopt.org/userguide/coneprog.html#quadratic-cone-programs
"""


def safe_cholesky(cov_matrix, min_eigenvalue=1e-4):
    """
    Compute Cholesky decomposition with regularization if needed.
    """
    cov_matrix = np.asarray(cov_matrix)

    # Check symmetry
    if not np.allclose(cov_matrix, cov_matrix.T):
        cov_matrix = (cov_matrix + cov_matrix.T) / 2

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov_matrix)

    regularized = False
    if eigenvalues.min() < min_eigenvalue:
        regularized = True
        # More aggressive regularization
        reg_amount = min_eigenvalue - eigenvalues.min() + 1e-3  # Add buffer
        cov_matrix_reg = cov_matrix + reg_amount * np.eye(len(cov_matrix))
    else:
        cov_matrix_reg = cov_matrix

    try:
        L = np.linalg.cholesky(cov_matrix_reg)
        return L, regularized
    except np.linalg.LinAlgError:
        # If still fails, use even more aggressive regularization
        reg_amount = 1e-2
        cov_matrix_reg = cov_matrix + reg_amount * np.eye(len(cov_matrix))
        L = np.linalg.cholesky(cov_matrix_reg)
        return L, True


def solve_bayesqp_with_fallback(
    bayesqp_object,
    x_k,
    delta_f=0.2,
    delta_c=0.2,
    ignore_crossterms=False,
    rho=100.0,
):
    """
    Solve BayeSQP subproblem (BSUB) with fallback to slacked version.
    """

    # standard BSUB
    bayesqp_object._print("Attempting standard BayeSQP subproblem...", msg_type="debug")

    try:
        result, timing = solve_bayesqp_subproblem(
            bayesqp_object=bayesqp_object,
            x_k=x_k,
            delta_f=delta_f,
            delta_c=delta_c,
            ignore_crossterms=ignore_crossterms,
        )

        if result is not None and result["status"] == "optimal":
            bayesqp_object._print("✓ Standard BSUB succeeded", msg_type="debug")
            result["solver_strategy"] = "BSUB"
            return result, timing
        else:
            bayesqp_object._print("✗ Standard BSUB failed", msg_type="warning")

    except Exception as e:
        bayesqp_object._print(f"✗ Standard BSUB crashed: {e}", msg_type="warning")

    # slacked BSUB
    bayesqp_object._print(
        f"Falling back to slacked BSUB with rho={rho}...",
        msg_type="warning",
    )

    try:
        result, timing = solve_bayesqp_subproblem_with_slacks(
            bayesqp_object=bayesqp_object,
            x_k=x_k,
            delta_f=delta_f,
            delta_c=delta_c,
            ignore_crossterms=ignore_crossterms,
            rho=rho,
        )

        if result is not None and result["status"] == "optimal":
            bayesqp_object._print("✓ Slacked BSUB succeeded", msg_type="debug")
            return result, timing
        else:
            bayesqp_object._print("✗ Slacked BSUB failed", msg_type="warning")

    except Exception as e:
        bayesqp_object._print(f"✗ Slacked BSUB crashed: {e}", msg_type="warning")

    # safe fallback
    bayesqp_object._print(
        "Also slacked BSUB failed, returning safe fallback", msg_type="error"
    )

    n = bayesqp_object.obj_model.D
    m = bayesqp_object.objective_function.num_constraints
    result = create_failed_bayesqp_result(n, m)
    result["solver_strategy"] = "fallback_failed"
    return result, {"gp_inference_time": 0.0, "sqp_time": 0.0}


def solve_bayesqp_subproblem_with_slacks(
    bayesqp_object,
    x_k,
    delta_f=0.1,
    delta_c=0.1,
    ignore_crossterms=False,
    rho=100.0,
):
    """
    Solve BayeSQPs uncertainty-aware subproblem with slack variables.
    """

    found_feasible = bayesqp_object.state.best_feasible

    if not found_feasible:
        delta_f = 0.5
        bayesqp_object._print(
            f"Relaxing delta_f to {delta_f} for feasibility.", msg_type="debug"
        )

    # Problem dimensions
    n = bayesqp_object.obj_model.D  # dimension of p
    m = bayesqp_object.objective_function.num_constraints  # number of constraints

    # Get predictions at current point x_k
    t0 = time.time()
    mu_f, _, mu_cs, _ = bayesqp_object.evaluate_point(x_k)
    mu_grad_f, _, mu_grad_cs, _ = bayesqp_object.evaluate_gradient(x_k)
    gp_inference_time = time.time() - t0

    # Compute confidence level quantiles
    q_1_delta_f = float(np.sqrt(2) * scipy.special.erfinv(2 * (1 - delta_f) - 1))
    q_1_delta_c = float(np.sqrt(2) * scipy.special.erfinv(2 * (1 - delta_c) - 1))

    bayesqp_object._print(
        f"Quantile factor for objective (1-δ_f): {q_1_delta_f}",
        msg_type="debug",
    )
    bayesqp_object._print(
        f"Quantile factor for constraints (1-δ_c): {q_1_delta_c}",
        msg_type="debug",
    )
    bayesqp_object._print(f"Current μ_f: {float(mu_f)}", msg_type="debug")
    bayesqp_object._print(f"Current μ_c: {[float(c) for c in mu_cs]}", msg_type="debug")

    # Variables: [p (n), b_f (1), b_{c_1}...b_{c_m} (m), s_1...s_m (m)]
    # Total: n + 1 + m + m = n + 1 + 2m variables
    full_dim = n + 1 + 2 * m

    # Indices for variables
    p_idx = slice(0, n)  # p: indices 0 to n-1
    b_f_idx = n  # b_f: index n
    b_c_idx = slice(n + 1, n + 1 + m)  # b_{c_i}: indices n+1 to n+m
    s_idx = slice(n + 1 + m, n + 1 + 2 * m)  # s_i: indices n+1+m to n+2m

    # Set up quadratic objective matrix P
    P = np.zeros((full_dim, full_dim))
    P[p_idx, p_idx] = bayesqp_object.H_k
    P = matrix(P)

    # Set up linear objective vector q
    q = np.zeros(full_dim)
    q[p_idx] = mu_grad_f
    q[b_f_idx] = q_1_delta_f
    q[s_idx] = rho
    q = matrix(q)

    # Linear inequality constraints: G x leq h
    G_list = []
    h_list = []

    # Relaxed constraints
    G_linear_constraints = np.zeros((m, full_dim))
    h_linear_constraints = np.zeros(m)

    for i in range(m):
        G_linear_constraints[i, p_idx] = -mu_grad_cs[i]
        G_linear_constraints[i, n + 1 + i] = q_1_delta_c
        G_linear_constraints[i, n + 1 + m + i] = -1.0
        h_linear_constraints[i] = float(mu_cs[i])

    G_list.append(G_linear_constraints)
    h_list.append(h_linear_constraints)

    G_nonneg = np.zeros((1 + 2 * m, full_dim))
    h_nonneg = np.zeros(1 + 2 * m)

    G_nonneg[0, b_f_idx] = -1.0
    for i in range(m):
        G_nonneg[1 + i, n + 1 + i] = -1.0
        G_nonneg[1 + m + i, n + 1 + m + i] = -1.0

    G_list.append(G_nonneg)
    h_list.append(h_nonneg)

    # Box constraints: 0 leq x_k + p leq 1
    G_lower = np.zeros((n, full_dim))
    G_lower[:, p_idx] = -np.eye(n)
    h_lower = x_k.numpy().flatten()

    G_upper = np.zeros((n, full_dim))
    G_upper[:, p_idx] = np.eye(n)
    h_upper = 1 - x_k.numpy().flatten()

    G_list.extend([G_lower, G_upper])
    h_list.extend([h_lower, h_upper])

    # Set up cone dimensions
    num_linear = m + (1 + 2 * m) + 2 * n  # constraints + nonneg + box constraints
    dims = {"l": num_linear, "q": [], "s": []}

    # Eval model for SOC constraint
    t0 = time.time()
    joint_cov_f = bayesqp_object.obj_model.joint_covariance(
        x_k, ignore_crossterms=ignore_crossterms
    ).numpy()
    gp_inference_time += time.time() - t0

    L_f, regularized_f = safe_cholesky(joint_cov_f)
    if regularized_f:
        bayesqp_object._print(
            "Warning: Objective covariance matrix regularized", msg_type="warning"
        )

    # SOC constraint
    cone_size_f = n + 2
    dims["q"].append(cone_size_f)

    G_soc_f = np.zeros((cone_size_f, full_dim))
    G_soc_f[0, b_f_idx] = -1.0
    G_soc_f[1:, p_idx] = L_f[:, 1:]

    h_soc_f = np.zeros(cone_size_f)
    h_soc_f[0] = 0.0  
    h_soc_f[1:] = -L_f[:, 0] 

    G_list.append(G_soc_f)
    h_list.append(h_soc_f)

    # Second-order cone constraints
    for i in range(m):
        t0 = time.time()
        joint_cov_ci = (
            bayesqp_object.constraint_models[i]
            .joint_covariance(x_k, ignore_crossterms=ignore_crossterms)
            .numpy()
        )
        gp_inference_time += time.time() - t0

        L_ci, regularized_ci = safe_cholesky(joint_cov_ci)
        if regularized_ci:
            bayesqp_object._print(
                f"Warning: Constraint {i} covariance matrix regularized",
                msg_type="warning",
            )

        cone_size_ci = n + 2
        dims["q"].append(cone_size_ci)

        G_soc_ci = np.zeros((cone_size_ci, full_dim))
        G_soc_ci[0, n + 1 + i] = -1.0  # -b_{c_i}
        G_soc_ci[1:, p_idx] = L_ci[:, 1:]  # L_{c_i}[:, 1:] * p

        h_soc_ci = np.zeros(cone_size_ci)
        h_soc_ci[0] = 0.0  # 0
        h_soc_ci[1:] = -L_ci[:, 0]  # -L_{c_i}[:, 0]

        G_list.append(G_soc_ci)
        h_list.append(h_soc_ci)

    # Combine all constraint matrices
    G = matrix(np.vstack(G_list))
    h = matrix(np.hstack(h_list))

    bayesqp_object._print(
        f"Slacked problem dimensions: n={n}, m={m}, total_vars={full_dim}",
        msg_type="debug",
    )
    bayesqp_object._print(f"G matrix shape: {G.size}", msg_type="debug")
    bayesqp_object._print(f"h vector shape: {h.size}", msg_type="debug")
    bayesqp_object._print(f"Cone dimensions: {dims}", msg_type="debug")

    # Solver settings
    solvers.options.clear()
    solvers.options["show_progress"] = False
    solvers.options["maxiters"] = 100000
    solvers.options["abstol"] = 1e-7
    solvers.options["reltol"] = 1e-7
    solvers.options["feastol"] = 1e-7

    try:
        t0 = time.time()
        result = solvers.coneqp(P=P, q=q, G=G, h=h, dims=dims)
        sqp_time = time.time() - t0

        bayesqp_object._print(
            f"Slacked solver status: {result['status']}", msg_type="debug"
        )

        if result["status"] != "optimal":
            bayesqp_object._print(
                f"Warning: Slacked BayeSQP status is {result['status']}",
                msg_type="warning",
            )
            bayesqp_object._print(
                f"Primal infeasibility: {result.get('primal infeasibility', 'N/A')}",
                msg_type="debug",
            )
            bayesqp_object._print(
                f"Dual infeasibility: {result.get('dual infeasibility', 'N/A')}",
                msg_type="debug",
            )
            return create_failed_bayesqp_result(n, m), {
                "gp_inference_time": gp_inference_time,
                "sqp_time": sqp_time,
            }

        # Extract solution
        x_sol = np.array(result["x"]).flatten()
        p_sol = x_sol[p_idx]
        b_f_sol = x_sol[b_f_idx]
        b_c_sol = x_sol[b_c_idx]
        s_sol = x_sol[s_idx]

        # Extract Lagrange multipliers
        lambda_c = extract_bayesqp_multipliers(result, dims, m, n)

        # Analyze slack usage
        active_slacks = np.where(s_sol > 1e-6)[0]
        total_slack_penalty = rho * np.sum(s_sol)

        if len(active_slacks) > 0:
            bayesqp_object._print(
                f"Active slack variables (constraints {active_slacks}):",
                msg_type="warning",
            )
            for idx in active_slacks:
                bayesqp_object._print(
                    f"  s_{idx} = {s_sol[idx]:.6f}", msg_type="warning"
                )
            bayesqp_object._print(
                f"Total slack penalty: {total_slack_penalty:.6f}", msg_type="warning"
            )
        else:
            bayesqp_object._print(
                "No slack variables active - original constraints satisfied",
                msg_type="debug",
            )

        predicted_constraints_original = []
        predicted_constraints_relaxed = []
        bayesqp_object._print("Constraint analysis:", msg_type="debug")
        # Handle potential array conversion issues
        for i in range(m):
            # Safely extract scalar values
            orig_val = mu_cs[i] + mu_grad_cs[i] @ p_sol
            relax_val = orig_val + s_sol[i]
            predicted_constraints_original.append(orig_val)
            predicted_constraints_relaxed.append(relax_val)
            slack_val = s_sol[i]

            # Convert to scalar if needed
            if hasattr(orig_val, "item"):
                orig_val = orig_val.item()
            if hasattr(relax_val, "item"):
                relax_val = relax_val.item()
            if hasattr(slack_val, "item"):
                slack_val = slack_val.item()

            bayesqp_object._print(
                f"  Constraint {i}: original = {float(orig_val):.6f}, "
                f"relaxed = {float(relax_val):.6f}, slack = {float(slack_val):.6f}",
                msg_type="debug",
            )

        bayesqp_object._print(
            f"Original constraints ≥ 0: {np.all(np.asarray(predicted_constraints_original) >= -1e-6)}",
            msg_type="debug",
        )
        bayesqp_object._print(
            f"Relaxed constraints ≥ 0: {np.all(np.asarray(predicted_constraints_relaxed) >= -1e-6)}",
            msg_type="debug",
        )

        # Compute objective value
        obj_quadratic = 0.5 * p_sol.T @ bayesqp_object.H_k @ p_sol
        obj_linear = np.dot(mu_grad_f, p_sol)
        obj_constant = float(mu_f)
        obj_uncertainty = q_1_delta_f * b_f_sol
        total_obj_original = obj_quadratic + obj_linear + obj_constant + obj_uncertainty
        total_obj_with_penalty = total_obj_original + total_slack_penalty

        bayesqp_object._print("Objective breakdown:", msg_type="debug")
        bayesqp_object._print(
            f"  Quadratic: (1/2)p^T H p = {obj_quadratic:.6f}",
            msg_type="debug",
        )
        bayesqp_object._print(
            f"  Linear: μ_{{∇f}}^T p = {obj_linear:.6f}",
            msg_type="debug",
        )
        bayesqp_object._print(
            f"  Constant: μ_f = {obj_constant:.6f}",
            msg_type="debug",
        )
        bayesqp_object._print(
            f"  Uncertainty: q_{{1-δ_f}} b_f = {obj_uncertainty:.6f}",
            msg_type="debug",
        )
        bayesqp_object._print(
            f"  Slack penalty: rho * Σ s_i = {total_slack_penalty:.6f}",
            msg_type="debug",
        )
        bayesqp_object._print(
            f"  Total (original): {total_obj_original:.6f}",
            msg_type="debug",
        )
        bayesqp_object._print(
            f"  Total (with penalty): {total_obj_with_penalty:.6f}",
            msg_type="debug",
        )

        return {
            "p": p_sol,
            "status": result["status"],
            "full_solution": x_sol,
            "auxiliary_vars": {
                "b_f": b_f_sol,
                "b_c": b_c_sol,
                "slack": s_sol,
            },
            "lambda_c": lambda_c,
            "predicted_constraints": predicted_constraints_original,
            "predicted_constraints_relaxed": predicted_constraints_relaxed,
            "objective_value": total_obj_original,
            "objective_value_with_penalty": total_obj_with_penalty,
            "slack_penalty": total_slack_penalty,
            "constraint_violations": s_sol,
            "solver_strategy": "bayesqp_slacked",
        }, {"gp_inference_time": gp_inference_time, "sqp_time": sqp_time}

    except Exception as e:
        bayesqp_object._print(
            f"Error in slacked BayeSQP solver: {e}", msg_type="warning"
        )
        import traceback

        bayesqp_object._print(f"Traceback: {traceback.format_exc()}", msg_type="debug")
        return create_failed_bayesqp_result(n, m), {
            "gp_inference_time": gp_inference_time,
            "sqp_time": 0,
        }


def create_failed_bayesqp_result(n, m):
    """
    Create a valid result structure for failed BayeSQP optimization attempts.
    """
    return {
        "p": np.zeros(n),
        "status": "failed",
        "full_solution": np.zeros(n + 1 + 2 * m),
        "auxiliary_vars": {
            "b_f": 0.0,
            "b_c": np.zeros(m),
            "slack": np.zeros(m),
        },
        "lambda_c": np.zeros(m),
        "predicted_constraints": np.zeros(m),
        "predicted_constraints_relaxed": np.zeros(m),
        "objective_value": float("inf"),
        "objective_value_with_penalty": float("inf"),
        "slack_penalty": 0.0,
        "constraint_violations": np.zeros(m),
        "solver_strategy": "failed",
    }


def solve_bayesqp_subproblem(
    bayesqp_object,
    x_k,
    delta_f=0.1,
    delta_c=0.1,
    ignore_crossterms=False,
    use_cvar=False,
):
    """
    Solve the BayeSQP uncertainty-aware subproblem (BSUB).
    """

    found_feasible = bayesqp_object.state.best_feasible

    if not found_feasible:
        delta_f = 0.5
        bayesqp_object._print(
            f"Relaxing delta_f to {delta_f} for feasibility.",
            msg_type="debug",
        )

    # Problem dimensions
    n = bayesqp_object.obj_model.D  # dimension of p
    m = bayesqp_object.objective_function.num_constraints  # number of constraints

    # Get predictions at current point x_t
    t0 = time.time()
    mu_f, _, mu_cs, _ = bayesqp_object.evaluate_point(x_k)
    mu_grad_f, _, mu_grad_cs, _ = bayesqp_object.evaluate_gradient(x_k)
    gp_inference_time = time.time() - t0

    q_1_delta_f = float(np.sqrt(2) * scipy.special.erfinv(2 * (1 - delta_f) - 1))
    q_1_delta_c = float(np.sqrt(2) * scipy.special.erfinv(2 * (1 - delta_c) - 1))

    bayesqp_object._print(
        f"Quantile factor for objective (1-δ_f): {q_1_delta_f}",
        msg_type="debug",
    )
    bayesqp_object._print(
        f"Quantile factor for constraints (1-δ_c): {q_1_delta_c}",
        msg_type="debug",
    )
    bayesqp_object._print(
        f"Current μ_f: {float(mu_f)}",
        msg_type="debug",
    )
    bayesqp_object._print(
        f"Current μ_c: {[float(c) for c in mu_cs]}",
        msg_type="debug",
    )

    # Variables: [p (n vars), b_f (1 var), b_{c_1}, ..., b_{c_m} (m vars)]
    # Total: n + 1 + m variables
    full_dim = n + 1 + m

    # Indices for variables
    p_idx = slice(0, n)  # p: indices 0 to n-1
    b_f_idx = n  # b_f: index n
    b_c_idx = slice(n + 1, n + 1 + m)  # b_{c_i}: indices n+1 to n+m

    # Set up quadratic objective matrix P
    P = np.zeros((full_dim, full_dim))
    P[p_idx, p_idx] = bayesqp_object.H_k  # (1/2) p^T H_t p term
    P = matrix(P)

    # Set up linear objective vector q
    q = np.zeros(full_dim)
    q[p_idx] = mu_grad_f  # μ_{∇f}^T p term
    q[b_f_idx] = q_1_delta_f  # q_{1-δ_f} b_f term
    q = matrix(q)

    # Linear inequality constraints: G x leq h
    G_list = []
    h_list = []

    # "standard" constraints
    G_linear_constraints = np.zeros((m, full_dim))
    h_linear_constraints = np.zeros(m)

    for i in range(m):
        G_linear_constraints[i, p_idx] = -mu_grad_cs[i]
        G_linear_constraints[i, n + 1 + i] = q_1_delta_c
        h_linear_constraints[i] = float(mu_cs[i])

    G_list.append(G_linear_constraints)
    h_list.append(h_linear_constraints)

    # Non-negativity constraints for b_f and b_{c_i}
    G_nonneg = np.zeros((1 + m, full_dim))
    h_nonneg = np.zeros(1 + m)

    G_nonneg[0, b_f_idx] = -1.0
    for i in range(m):
        G_nonneg[1 + i, n + 1 + i] = -1.0

    G_list.append(G_nonneg)
    h_list.append(h_nonneg)

    # Box constraints: 0 ≤ x_t + p ≤ 1
    G_lower = np.zeros((n, full_dim))
    G_lower[:, p_idx] = -np.eye(n)
    h_lower = x_k.numpy().flatten()

    G_upper = np.zeros((n, full_dim))
    G_upper[:, p_idx] = np.eye(n)
    h_upper = 1 - x_k.numpy().flatten()

    G_list.extend([G_lower, G_upper])
    h_list.extend([h_lower, h_upper])

    # Set up cone dimensions
    num_linear = m + (1 + m) + 2 * n
    dims = {"l": num_linear, "q": [], "s": []}

    t0 = time.time()
    joint_cov_f = bayesqp_object.obj_model.joint_covariance(
        x_k, ignore_crossterms=ignore_crossterms
    ).numpy()
    gp_inference_time += time.time() - t0

    L_f, regularized_f = safe_cholesky(joint_cov_f)
    if regularized_f:
        bayesqp_object._print(
            "Warning: Objective covariance matrix regularized",
            msg_type="warning",
        )

    cone_size_f = n + 2
    dims["q"].append(cone_size_f)

    G_soc_f = np.zeros((cone_size_f, full_dim))
    h_soc_f = np.zeros(cone_size_f)

    G_soc_f[0, b_f_idx] = -1.0
    G_soc_f[1:, p_idx] = L_f[:, 1:]
    h_soc_f[0] = 0.0
    h_soc_f[1:] = -L_f[:, 0]

    G_list.append(G_soc_f)
    h_list.append(h_soc_f)

    for i in range(m):
        t0 = time.time()
        joint_cov_ci = (
            bayesqp_object.constraint_models[i]
            .joint_covariance(x_k, ignore_crossterms=ignore_crossterms)
            .numpy()
        )
        gp_inference_time += time.time() - t0

        L_ci, regularized_ci = safe_cholesky(joint_cov_ci)
        if regularized_ci:
            bayesqp_object._print(
                f"Warning: Constraint {i} covariance matrix regularized",
                msg_type="warning",
            )

        cone_size_ci = n + 2
        dims["q"].append(cone_size_ci)
        G_soc_ci = np.zeros((cone_size_ci, full_dim))
        h_soc_ci = np.zeros(cone_size_ci)

        G_soc_ci[0, n + 1 + i] = -1.0
        G_soc_ci[1:, p_idx] = L_ci[:, 1:]
        h_soc_ci[0] = 0.0
        h_soc_ci[1:] = -L_ci[:, 0]

        G_list.append(G_soc_ci)
        h_list.append(h_soc_ci)

    # Combine all constraint matrices
    G = matrix(np.vstack(G_list))
    h = matrix(np.hstack(h_list))

    bayesqp_object._print(
        f"Problem dimensions: n={n}, m={m}, total_vars={full_dim}", msg_type="debug"
    )
    bayesqp_object._print(f"G matrix shape: {G.size}", msg_type="debug")
    bayesqp_object._print(f"h vector shape: {h.size}", msg_type="debug")
    bayesqp_object._print(f"Cone dimensions: {dims}", msg_type="debug")

    # Solver settings
    solvers.options.clear()
    solvers.options["show_progress"] = False
    solvers.options["maxiters"] = 100000
    solvers.options["abstol"] = 1e-7
    solvers.options["reltol"] = 1e-7
    solvers.options["feastol"] = 1e-7

    try:
        t0 = time.time()
        result = solvers.coneqp(P=P, q=q, G=G, h=h, dims=dims)
        sqp_time = time.time() - t0

        bayesqp_object._print(f"Solver status: {result['status']}", msg_type="debug")

        if result["status"] != "optimal":
            bayesqp_object._print(
                f"Warning: BayeSQP subproblem status is {result['status']}",
                msg_type="warning",
            )
            bayesqp_object._print(
                f"Primal infeasibility: {result.get('primal infeasibility', 'N/A')}",
                msg_type="debug",
            )
            bayesqp_object._print(
                f"Dual infeasibility: {result.get('dual infeasibility', 'N/A')}",
                msg_type="debug",
            )
            return None, {"gp_inference_time": gp_inference_time, "sqp_time": sqp_time}

        # Extract solution
        x_sol = np.array(result["x"]).flatten()
        p_sol = x_sol[p_idx]
        b_f_sol = x_sol[b_f_idx]
        b_c_sol = x_sol[b_c_idx]

        # Extract Lagrange multipliers
        lambda_c = extract_bayesqp_multipliers(result, dims, m, n)

        # Verify constraint satisfaction
        predicted_constraints = []
        for i, mu_c in enumerate(mu_cs):
            pred_c_i = mu_c + mu_grad_cs[i] @ p_sol
            predicted_constraints.append(pred_c_i)
            bayesqp_object._print(
                f"Predicted constraint values μ_c + μ_gradc^T p: {pred_c_i.item():.6f}",
                msg_type="debug",
            )
        bayesqp_object._print(
            f"All constraints ≥ 0: {np.all(np.asarray(predicted_constraints) >= -1e-6)}",
            msg_type="debug",
        )

        # Compute objective value
        obj_quadratic = 0.5 * p_sol.T @ bayesqp_object.H_k @ p_sol
        obj_linear = np.dot(mu_grad_f, p_sol)
        obj_constant = float(mu_f)
        obj_uncertainty = q_1_delta_f * b_f_sol
        total_obj = obj_quadratic + obj_linear + obj_constant + obj_uncertainty

        bayesqp_object._print("Objective breakdown:", msg_type="debug")
        bayesqp_object._print(
            f"  Quadratic: (1/2)p^T H p = {obj_quadratic:.6f}", msg_type="debug"
        )
        bayesqp_object._print(
            f"  Linear: mu_grad_f^T p = {obj_linear:.6f}", msg_type="debug"
        )
        bayesqp_object._print(
            f"  Constant: mu_f = {obj_constant:.6f}", msg_type="debug"
        )
        bayesqp_object._print(
            f"  Uncertainty: q_{1 - delta_f} b_f = {obj_uncertainty:.6f}",
            msg_type="debug",
        )
        bayesqp_object._print(f"  Total: {total_obj:.6f}", msg_type="debug")

        return {
            "p": p_sol,
            "status": result["status"],
            "full_solution": x_sol,
            "auxiliary_vars": {
                "b_f": b_f_sol,
                "b_c": b_c_sol,
            },
            "lambda_c": lambda_c,
            "predicted_constraints": predicted_constraints,
            "objective_value": total_obj,
        }, {"gp_inference_time": gp_inference_time, "sqp_time": sqp_time}

    except Exception as e:
        bayesqp_object._print(
            f"Error in BayeSQP subproblem solver: {e}", msg_type="warning"
        )
        bayesqp_object._print(f"Traceback: {traceback.format_exc()}", msg_type="debug")
        return None, {"gp_inference_time": gp_inference_time, "sqp_time": 0}


def extract_bayesqp_multipliers(result, dims, m, n):
    """
    Extract Lagrange multipliers for the linear constraints in BayeSQP subproblem.
    """

    if result["status"] != "optimal":
        print("Warning: Solution not optimal, dual variables may be unreliable")
        return np.zeros(m)

    # Extract dual variables for linear constraints G x leq h
    z = np.array(result["z"]).flatten()
    lambda_c = z[:m]
    return lambda_c


def solve_bayesqp_unconstrained(
    bayesqp_object,
    x_k,
    delta_f=0.2,
    ignore_crossterms=False,
):
    """
    Solve the unconstrained BayeSQP subproblem.
    """

    # Problem dimensions
    n = bayesqp_object.obj_model.D  # dimension of p

    try:
        # Get predictions at current point x_k
        t0 = time.time()
        mu_f, _, _, _ = bayesqp_object.evaluate_point(x_k)
        mu_grad_f, _, _, _ = bayesqp_object.evaluate_gradient(x_k)
        gp_inference_time = time.time() - t0

        # Compute confidence level quantile
        q_1_delta_f = float(np.sqrt(2) * scipy.special.erfinv(2 * (1 - delta_f) - 1))

        bayesqp_object._print(
            f"Quantile factor for objective (1-δ_f): {q_1_delta_f}", msg_type="debug"
        )

        bayesqp_object._print(f"Current μ_f: {mu_f}", msg_type="debug")

        # Variables: [p (n vars), b_f (1 var)]
        # Total: n + 1 variables
        full_dim = n + 1

        # Indices for variables
        p_idx = slice(0, n)  # p: indices 0 to n-1
        b_f_idx = n  # b_f: index n

        # Set up quadratic objective matrix P
        P = np.zeros((full_dim, full_dim))
        P[p_idx, p_idx] = bayesqp_object.H_k
        P = matrix(P)

        # Set up linear objective vector q
        q = np.zeros(full_dim)
        q[p_idx] = mu_grad_f
        q[b_f_idx] = q_1_delta_f
        q = matrix(q)

        # Linear inequality constraints
        G_list = []
        h_list = []

        # Non-negativity constraint for b_f
        G_nonneg = np.zeros((1, full_dim))
        h_nonneg = np.zeros(1)

        G_nonneg[0, b_f_idx] = -1.0

        G_list.append(G_nonneg)
        h_list.append(h_nonneg)

        # Box constraints to ensure that we remain in [0, 1]^D
        G_lower = np.zeros((n, full_dim))
        G_lower[:, p_idx] = -np.eye(n)
        h_lower = x_k.numpy().flatten()

        G_upper = np.zeros((n, full_dim))
        G_upper[:, p_idx] = np.eye(n)
        h_upper = 1 - x_k.numpy().flatten()

        G_list.extend([G_lower, G_upper])
        h_list.extend([h_lower, h_upper])

        # Set up cone dimensions
        num_linear = 1 + 2 * n  # nonneg + box constraints
        dims = {"l": num_linear, "q": [], "s": []}

        t0 = time.time()
        joint_cov_f = bayesqp_object.obj_model.joint_covariance(
            x_k, ignore_crossterms=ignore_crossterms
        ).numpy()
        gp_inference_time += time.time() - t0

        L_f, regularized_f = safe_cholesky(joint_cov_f)
        if regularized_f:
            bayesqp_object._print(
                "Warning: Objective covariance matrix regularized", msg_type="warning"
            )

        # SOC constraint
        cone_size_f = n + 2
        dims["q"].append(cone_size_f)

        G_soc_f = np.zeros((cone_size_f, full_dim))
        G_soc_f[0, b_f_idx] = -1.0
        G_soc_f[1:, p_idx] = L_f[:, 1:]

        h_soc_f = np.zeros(cone_size_f)
        h_soc_f[0] = 0.0
        h_soc_f[1:] = -L_f[:, 0]

        G_list.append(G_soc_f)
        h_list.append(h_soc_f)

        # Combine all constraint matrices
        G = matrix(np.vstack(G_list))
        h = matrix(np.hstack(h_list))

        bayesqp_object._print(
            f"Unconstrained problem dimensions: n={n}, total_vars={full_dim}",
            msg_type="debug",
        )
        bayesqp_object._print(f"G matrix shape: {G.size}", msg_type="debug")
        bayesqp_object._print(f"h vector shape: {h.size}", msg_type="debug")
        bayesqp_object._print(f"Cone dimensions: {dims}", msg_type="debug")

        # Solver settings
        solvers.options.clear()
        solvers.options["show_progress"] = False
        solvers.options["maxiters"] = 100000
        solvers.options["abstol"] = 1e-7
        solvers.options["reltol"] = 1e-7
        solvers.options["feastol"] = 1e-7

        try:
            t0 = time.time()
            result = solvers.coneqp(P=P, q=q, G=G, h=h, dims=dims)
            sqp_time = time.time() - t0

            bayesqp_object._print(
                f"Solver status: {result['status']}", msg_type="debug"
            )

            # Extract solution
            x_sol = np.array(result["x"]).flatten()
            p_sol = x_sol[p_idx]
            b_f_sol = x_sol[b_f_idx]

            # Compute objective value, TODO: maybe only track for debugging for runtime
            obj_quadratic = 0.5 * p_sol.T @ bayesqp_object.H_k @ p_sol
            obj_linear = np.dot(mu_grad_f, p_sol)
            obj_constant = float(mu_f)
            obj_uncertainty = q_1_delta_f * b_f_sol
            total_obj = obj_quadratic + obj_linear + obj_constant + obj_uncertainty

            bayesqp_object._print("Objective breakdown:", msg_type="debug")
            bayesqp_object._print(
                f"  Quadratic: (1/2)p^T H p = {obj_quadratic:.6f}", msg_type="debug"
            )
            bayesqp_object._print(
                f"  Linear: μ_{{∇f}}^T p = {obj_linear:.6f}", msg_type="debug"
            )
            bayesqp_object._print(
                f"  Constant: μ_f = {obj_constant:.6f}", msg_type="debug"
            )
            bayesqp_object._print(
                f"  Uncertainty: q_{{1-δ_f}} b_f = {obj_uncertainty:.6f}",
                msg_type="debug",
            )
            bayesqp_object._print(f"  Total: {total_obj:.6f}", msg_type="debug")

            return {
                "p": p_sol,
                "status": result["status"],
                "full_solution": x_sol,
                "auxiliary_vars": {
                    "b_f": b_f_sol,
                },
                "lambda_c": np.array([]),  # Empty for unconstrained
                "objective_value": total_obj,
                "solver_strategy": "bayesqp_unconstrained",
            }, {"gp_inference_time": gp_inference_time, "sqp_time": sqp_time}

        except Exception as e:
            bayesqp_object._print(
                f"Error in unconstrained BayeSQP solver: {e}",
                msg_type="warning",
            )
            bayesqp_object._print(
                f"Traceback: {traceback.format_exc()}", msg_type="debug"
            )
            return None

    except Exception as e:
        bayesqp_object._print(
            f"Critical error in unconstrained BayeSQP solver: {e}",
            msg_type="error",
        )
        bayesqp_object._print(f"Traceback: {traceback.format_exc()}", msg_type="debug")
        return None
