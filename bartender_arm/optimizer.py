"""
optimizer.py — Constrained trajectory optimization for mixing motion.

No ROS2 dependencies. Pure numpy + scipy.

Optimization Problem
--------------------
Find Fourier trajectory coefficients x ∈ R^36 that minimize:

    J(x) = (1/N) Σ_t [ w₁ · ||τ(t)||² - w₂ · ||a_ee_lin(t)||² ]

Subject to:
    |q_i(t)|    ≤ q_max_i           ∀ i, t   (joint position limits)
    |q̇_i(t)|   ≤ v_max_i           ∀ i, t   (velocity limits)
    |τ_i(t)|    ≤ τ_max_i           ∀ i, t   (torque limits)
    Σ_ik x²     ≥ A_min              (non-trivial motion)

The negative w₂ term means we are simultaneously maximizing EE acceleration
while minimizing torque — the core engineering tradeoff.

Design choices
--------------
- Use N_opt=30 time points during optimization (fast evaluations)
- Use N_eval=300 points for final reporting and plots
- Use scipy.optimize.minimize with method='SLSQP'
- Constraints are expressed as inequality: g(x) >= 0
- Objective and constraint gradients are computed by scipy using
  finite differences (SLSQP supports this via default 'eps' parameter)
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint, OptimizeResult

from bartender_arm.kinematics import ee_acceleration, ee_position
from bartender_arm.dynamics import compute_torques
from bartender_arm.trajectory import make_trajectory, baseline_params


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def objective(x, t_arr, q0, f, params, weights, n_harmonics: int = 3) -> float:
    """
    Compute optimization objective J(x).

    J(x) = (1/N) Σ_t [ w_τ·||τ||² - w_a·(a_ee·n̂)² + w_d·||p_ee_perp - p_target_perp||² ]

    Rewards acceleration projected onto the shake axis n̂ (directional, not centripetal),
    and penalizes EE drift perpendicular to that axis.

    weights keys: 'weight_torque', 'weight_accel', 'weight_drift'
    params keys:  optimization.shake_axis  (unit vector, default [1,0,0])
    """
    w_tau   = float(weights.get('weight_torque', 1.0))
    w_acc   = float(weights.get('weight_accel',  0.5))
    w_drift = float(weights.get('weight_drift',  0.0))

    opt_cfg    = params['optimization']
    shake_axis = np.array(opt_cfg.get('shake_axis', [1.0, 0.0, 0.0]), dtype=float)
    shake_axis = shake_axis / np.linalg.norm(shake_axis)

    q0 = np.asarray(q0, dtype=float)
    p_target = ee_position(q0, params)

    traj = make_trajectory(x, f, t_arr, q0, n_harmonics=n_harmonics)
    N    = len(t_arr)

    J_torque = 0.0
    J_accel  = 0.0
    J_drift  = 0.0

    for t_idx in range(N):
        q_t     = traj['q'][t_idx]
        qdot_t  = traj['qdot'][t_idx]
        qddot_t = traj['qddot'][t_idx]

        tau  = compute_torques(q_t, qdot_t, qddot_t, params)
        a_ee = ee_acceleration(q_t, qdot_t, qddot_t, params)

        J_torque += np.dot(tau, tau)
        # Directional: only reward acceleration along shake axis (blocks centripetal loophole)
        proj = np.dot(a_ee[:3], shake_axis)
        J_accel += proj * proj

        if w_drift > 0.0:
            p_ee  = ee_position(q_t, params)
            diff  = p_ee - p_target
            # Penalize drift perpendicular to shake axis
            drift = diff - np.dot(diff, shake_axis) * shake_axis
            J_drift += np.dot(drift, drift)

    J_torque /= N
    J_accel  /= N
    J_drift  /= N

    return w_tau * J_torque - w_acc * J_accel + w_drift * J_drift


def objective_and_components(x, t_arr, q0, f, params, weights,
                              n_harmonics: int = 3) -> tuple:
    """
    Like objective() but also returns the three component values.

    Returns
    -------
    J         : float — total objective
    J_torque  : float — mean squared torque norm
    J_accel   : float — mean squared directional EE acceleration
    J_drift   : float — mean squared EE drift perpendicular to shake axis
    """
    w_tau   = float(weights.get('weight_torque', 1.0))
    w_acc   = float(weights.get('weight_accel',  0.5))
    w_drift = float(weights.get('weight_drift',  0.0))

    opt_cfg    = params['optimization']
    shake_axis = np.array(opt_cfg.get('shake_axis', [1.0, 0.0, 0.0]), dtype=float)
    shake_axis = shake_axis / np.linalg.norm(shake_axis)

    q0 = np.asarray(q0, dtype=float)
    p_target = ee_position(q0, params)

    traj = make_trajectory(x, f, t_arr, q0, n_harmonics=n_harmonics)
    N    = len(t_arr)

    J_torque = 0.0
    J_accel  = 0.0
    J_drift  = 0.0

    for t_idx in range(N):
        q_t     = traj['q'][t_idx]
        qdot_t  = traj['qdot'][t_idx]
        qddot_t = traj['qddot'][t_idx]

        tau  = compute_torques(q_t, qdot_t, qddot_t, params)
        a_ee = ee_acceleration(q_t, qdot_t, qddot_t, params)

        J_torque += np.dot(tau, tau)
        proj = np.dot(a_ee[:3], shake_axis)
        J_accel += proj * proj

        if w_drift > 0.0:
            p_ee  = ee_position(q_t, params)
            diff  = p_ee - p_target
            drift = diff - np.dot(diff, shake_axis) * shake_axis
            J_drift += np.dot(drift, drift)

    J_torque /= N
    J_accel  /= N
    J_drift  /= N

    return w_tau * J_torque - w_acc * J_accel + w_drift * J_drift, J_torque, J_accel, J_drift


# ---------------------------------------------------------------------------
# Constraint generation
# ---------------------------------------------------------------------------

def make_constraints(t_arr, q0, f, params, n_harmonics: int = 3) -> list:
    """
    Build SLSQP inequality constraints: g(x) >= 0.

    Uses batched vector-valued constraint functions (one call per type per
    gradient step) instead of 721 separate scalar functions. This cuts
    trajectory evaluations from ~26,000 to ~4 per SLSQP iteration.

    Constraints:
        1. q_max[i] - q[t,i]                       >= 0   for all t, i  (upper pos limit)
        2. q[t,i]   - q_min[i]                     >= 0   for all t, i  (lower pos limit)
        3. tau_max[i] - |tau[t,i]|                 >= 0   for all t, i  (torque limit)
        4. v_max[i] - Σ_k kω√(a_ik²+b_ik²)        >= 0   for all i     (velocity, analytic)
        5. sum(x²)  - A_min                         >= 0                 (minimum motion)

    The velocity constraint uses the triangle-inequality upper bound on |qdot_i(t)|
    for ALL t — not a sampled check.  This is exact regardless of n_opt_points and
    avoids missed peaks when the highest harmonic (k=3 at f=1 Hz → 3 Hz) is
    undersampled by the optimization time grid.
    """
    robot   = params['robot']
    opt_cfg = params['optimization']

    jlimits = robot['joint_limits']
    jnames  = robot['joint_names']
    q_min   = np.array([jlimits[j][0] for j in jnames], dtype=float)
    q_max   = np.array([jlimits[j][1] for j in jnames], dtype=float)
    v_max   = np.array(robot['velocity_limits'], dtype=float)
    tau_max = np.array(robot['effort_limits'],   dtype=float)
    A_min   = float(opt_cfg.get('min_amplitude_sq', 0.01))

    safety  = float(opt_cfg.get('constraint_safety_margin', 1.0))
    v_max   = v_max   * safety
    tau_max = tau_max * safety

    N = len(t_arr)

    def _eval(x):
        """Compute traj + torques once, reused by all batched constraints."""
        traj = make_trajectory(x, f, t_arr, q0, n_harmonics=n_harmonics)
        torques = np.array([
            compute_torques(traj['q'][i], traj['qdot'][i], traj['qddot'][i], params)
            for i in range(N)
        ])
        return traj['q'], traj['qdot'], torques

    def pos_constraints(x):
        q, _, _ = _eval(x)
        upper = (q_max - q).ravel()  # q_max[j] - q[t,j] >= 0
        lower = (q - q_min).ravel()  # q[t,j] - q_min[j] >= 0
        return np.concatenate([upper, lower])

    def vel_constraints(x):
        # Analytical upper bound on peak |qdot_i(t)| via triangle inequality:
        #   max_t |qdot_i(t)| <= sum_k kω * sqrt(a_ik^2 + b_ik^2)
        # Exact for all t — no sampling error regardless of n_opt_points.
        K = n_harmonics
        a_v = x[:6 * K].reshape(6, K)
        b_v = x[6 * K:].reshape(6, K)
        omega = 2.0 * np.pi * f
        peak_v = np.array([
            sum(np.sqrt(a_v[i, k] ** 2 + b_v[i, k] ** 2) * (k + 1) * omega
                for k in range(K))
            for i in range(6)
        ])
        return v_max - peak_v  # >= 0 means feasible

    def torque_constraints(x):
        _, _, torques = _eval(x)
        return (tau_max - np.abs(torques)).ravel()  # tau_max[j] - |tau[t,j]| >= 0

    def min_amplitude(x):
        return np.dot(x, x) - A_min

    return [
        {'type': 'ineq', 'fun': pos_constraints},
        {'type': 'ineq', 'fun': vel_constraints},
        {'type': 'ineq', 'fun': torque_constraints},
        {'type': 'ineq', 'fun': min_amplitude},
    ]


def make_bounds(params, n_harmonics: int = 3, max_coeff: float = None):
    """
    Build variable bounds for SLSQP.

    Limits each Fourier coefficient to [-max_coeff, +max_coeff].

    Parameters
    ----------
    max_coeff : float, optional
        Override for the coefficient bound. If None, reads from
        params['optimization']['max_coeff_amplitude'].

    Returns
    -------
    list of (lb, ub) tuples, length 2*6*n_harmonics = 36
    """
    if max_coeff is None:
        max_coeff = float(params['optimization'].get('max_coeff_amplitude', 0.5))
    total = 2 * 6 * n_harmonics
    return [(-max_coeff, max_coeff)] * total


# ---------------------------------------------------------------------------
# Main optimization routine
# ---------------------------------------------------------------------------

def run_optimization(q0, f: float, t_arr_opt, params,
                     x0=None, n_harmonics: int = 3,
                     verbose: bool = True,
                     max_coeff: float = None) -> OptimizeResult:
    """
    Run constrained SLSQP optimization to find the best mixing trajectory.

    Parameters
    ----------
    q0         : (6,) center joint pose
    f          : float, trajectory frequency (Hz)
    t_arr_opt  : (N_opt,) time array for optimization (use N_opt=30 for speed)
    params     : dict
    x0         : (36,) initial guess, or None to use baseline
    n_harmonics: int
    verbose    : bool

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        result.x contains the optimal coefficient vector
        result.success indicates convergence
        result.fun is the final objective value
    """
    opt_cfg  = params['optimization']
    weights  = {
        'weight_torque': float(opt_cfg.get('weight_torque', 1.0)),
        'weight_accel':  float(opt_cfg.get('weight_accel',  0.5)),
        'weight_drift':  float(opt_cfg.get('weight_drift',  0.0)),
    }
    max_iter = int(opt_cfg.get('max_iter', 300))
    ftol     = float(opt_cfg.get('ftol', 1e-6))

    if x0 is None:
        x0 = baseline_params(n_joints=6, n_harmonics=n_harmonics)

    q0      = np.asarray(q0, dtype=float)
    t_arr   = np.asarray(t_arr_opt, dtype=float)
    bounds  = make_bounds(params, n_harmonics, max_coeff=max_coeff)

    # Clip x0 to bounds so SLSQP doesn't warn about out-of-bounds initial point.
    # This happens when x0=baseline_params has coefficients larger than max_coeff_amplitude.
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    x0 = np.clip(x0, lb, ub)

    constraints = make_constraints(t_arr, q0, f, params, n_harmonics)

    if verbose:
        N_t = len(t_arr)
        n_scalar = 2 * N_t * 6 + N_t * 6 + N_t * 6 + 1  # pos(2NJ) + vel(NJ) + torque(NJ) + min_amp
        print(f"Starting SLSQP optimization:")
        print(f"  Variables  : {len(x0)}")
        print(f"  Constraints: ~{n_scalar} scalar (4 batched vector fns)")
        print(f"  Time points: {len(t_arr)} (opt), {params['optimization']['n_eval_points']} (eval)")
        print(f"  Max iter   : {max_iter}")
        J0, J_tau0, J_acc0, J_drift0 = objective_and_components(
            x0, t_arr, q0, f, params, weights, n_harmonics)
        print(f"  Initial J  : {J0:.6f}  (J_torque={J_tau0:.4f}, J_accel={J_acc0:.4f}, J_drift={J_drift0:.4f})")

    def obj_fn(x):
        return objective(x, t_arr, q0, f, params, weights, n_harmonics)

    result = minimize(
        fun=obj_fn,
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={
            'maxiter': max_iter,
            'ftol':    ftol,
            'disp':    verbose,
        },
    )

    if verbose:
        J_opt, J_tau_opt, J_acc_opt, J_drift_opt = objective_and_components(
            result.x, t_arr, q0, f, params, weights, n_harmonics)
        print(f"\nOptimization {'converged' if result.success else 'did NOT converge'}:")
        print(f"  Final J    : {J_opt:.6f}  (J_torque={J_tau_opt:.4f}, J_accel={J_acc_opt:.4f}, J_drift={J_drift_opt:.4f})")
        print(f"  Message    : {result.message}")

    return result


# ---------------------------------------------------------------------------
# Multi-start optimization (escape local minima)
# ---------------------------------------------------------------------------

def multi_start_optimization(q0, f: float, t_arr_opt, params,
                              n_restarts: int = 8, n_harmonics: int = 3,
                              verbose: bool = True,
                              max_coeff: float = None,
                              rng_seed: int = 42) -> OptimizeResult:
    """
    Run SLSQP from n_restarts different starting points and return the best.

    Starting points:
        - Start 0: baseline_params (deterministic, gives reproducibility)
        - Starts 1..n_restarts-1: random coefficients uniformly in [-max_coeff, max_coeff]

    The best result is selected by lowest objective value among converged runs,
    falling back to the best non-converged run if none converged.

    Parameters
    ----------
    q0, f, t_arr_opt, params, n_harmonics, verbose, max_coeff
        Same as run_optimization().
    n_restarts : int
        Total number of starting points (including the baseline start).
    rng_seed : int
        Seed for reproducible random starts.

    Returns
    -------
    best_result : scipy.optimize.OptimizeResult
        Best result across all starts.  Also has attribute `start_index`
        indicating which start produced it.
    """
    if max_coeff is None:
        max_coeff = float(params['optimization'].get('max_coeff_amplitude', 0.20))

    rng   = np.random.default_rng(rng_seed)
    n_var = 2 * 6 * n_harmonics  # 36 for default settings

    best_result = None
    best_J      = np.inf

    for i in range(n_restarts):
        if i == 0:
            x0 = baseline_params(n_joints=6, n_harmonics=n_harmonics)
            label = "baseline"
        else:
            x0    = rng.uniform(-max_coeff, max_coeff, size=n_var)
            label = f"random-{i}"

        if verbose:
            print(f"\n{'='*60}")
            print(f"Multi-start {i+1}/{n_restarts}  (init: {label})")
            print(f"{'='*60}")

        result = run_optimization(
            q0=q0, f=f, t_arr_opt=t_arr_opt, params=params,
            x0=x0, n_harmonics=n_harmonics, verbose=verbose,
            max_coeff=max_coeff)
        result.start_index = i
        result.start_label = label

        if result.fun < best_J:
            best_J      = result.fun
            best_result = result

    if verbose:
        print(f"\n{'='*60}")
        print(f"Multi-start complete — best start: {best_result.start_label} "
              f"(J={best_J:.6f})")
        print(f"{'='*60}\n")

    return best_result


# ---------------------------------------------------------------------------
# Global optimization (Differential Evolution + SLSQP refinement)
# ---------------------------------------------------------------------------

def global_optimization(q0, f: float, t_arr_opt, params,
                        n_harmonics: int = 3,
                        verbose: bool = True,
                        max_coeff: float = None,
                        de_popsize: int = 5,
                        de_maxiter: int = 200,
                        rng_seed: int = 42) -> OptimizeResult:
    """
    Two-phase global optimizer: Differential Evolution then SLSQP refinement.

    Phase 1 — Differential Evolution explores the full search space without
    getting trapped in local minima.  Uses the same constraints as SLSQP.

    Phase 2 — SLSQP refines the DE solution to tight convergence.

    Parameters
    ----------
    q0, f, t_arr_opt, params, n_harmonics, verbose, max_coeff
        Same as run_optimization().
    de_popsize : int
        DE population multiplier (total population = de_popsize * n_vars).
        Higher = more thorough but slower.  Default 10.
    de_maxiter : int
        Maximum DE generations.  Default 500.
    rng_seed : int
        Seed for reproducibility.

    Returns
    -------
    result : scipy.optimize.OptimizeResult from the SLSQP refinement step.
    """
    if max_coeff is None:
        max_coeff = float(params['optimization'].get('max_coeff_amplitude', 0.20))

    opt_cfg = params['optimization']
    weights = {
        'weight_torque': float(opt_cfg.get('weight_torque', 1.0)),
        'weight_accel':  float(opt_cfg.get('weight_accel',  0.5)),
        'weight_drift':  float(opt_cfg.get('weight_drift',  0.0)),
    }

    q0    = np.asarray(q0, dtype=float)
    t_arr = np.asarray(t_arr_opt, dtype=float)
    bounds      = make_bounds(params, n_harmonics, max_coeff=max_coeff)
    constraints = make_constraints(t_arr, q0, f, params, n_harmonics)
    n_var       = 2 * 6 * n_harmonics

    def obj_fn(x):
        return objective(x, t_arr, q0, f, params, weights, n_harmonics)

    if verbose:
        print(f"Phase 1 — Differential Evolution")
        print(f"  Population : {de_popsize * n_var}  ({de_popsize} × {n_var} vars)")
        print(f"  Max generations: {de_maxiter}")

    # differential_evolution requires NonlinearConstraint objects, not SLSQP dicts
    de_constraints = [
        NonlinearConstraint(c['fun'], 0.0, np.inf) for c in constraints
    ]

    de_result = differential_evolution(
        func=obj_fn,
        bounds=bounds,
        constraints=de_constraints,
        maxiter=de_maxiter,
        popsize=de_popsize,
        seed=rng_seed,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
        disp=verbose,
    )

    if verbose:
        J_de, J_tau_de, J_acc_de, J_drift_de = objective_and_components(
            de_result.x, t_arr, q0, f, params, weights, n_harmonics)
        print(f"  DE result  : J={J_de:.6f}  (J_torque={J_tau_de:.4f}, J_accel={J_acc_de:.4f}, J_drift={J_drift_de:.4f})")
        print(f"\nPhase 2 — SLSQP refinement from DE solution")

    result = run_optimization(
        q0=q0, f=f, t_arr_opt=t_arr, params=params,
        x0=de_result.x, n_harmonics=n_harmonics,
        verbose=verbose, max_coeff=max_coeff)

    result.de_result = de_result

    if not result.success:
        # SLSQP linesearch failed — check whether the stopped point is still physically feasible.
        # If yes, keep it (it may be better than DE). If no, fall back to the DE solution.
        constr_check = check_constraints(result.x, t_arr, q0, f, params, n_harmonics)
        if constr_check['satisfied']:
            if verbose:
                print("  SLSQP didn't converge but result is feasible — keeping it.")
        else:
            if verbose:
                print("  SLSQP result is infeasible — falling back to DE solution.")
            de_result.de_result = de_result
            return de_result

    return result


# ---------------------------------------------------------------------------
# Constraint satisfaction check
# ---------------------------------------------------------------------------

def check_constraints(x, t_arr, q0, f, params, n_harmonics: int = 3) -> dict:
    """
    Check whether all physical constraints are satisfied for trajectory x.

    Returns
    -------
    dict with:
        satisfied : bool — True if all constraints pass
        violations : list of str — human-readable descriptions of violations
        margin : dict — worst-case margin per constraint type
    """
    robot   = params['robot']
    jlimits = robot['joint_limits']
    jnames  = robot['joint_names']
    q_min   = np.array([jlimits[j][0] for j in jnames], dtype=float)
    q_max   = np.array([jlimits[j][1] for j in jnames], dtype=float)
    v_max   = np.array(robot['velocity_limits'], dtype=float)
    tau_max = np.array(robot['effort_limits'],   dtype=float)

    traj = make_trajectory(x, f, t_arr, q0, n_harmonics=n_harmonics)
    N = len(t_arr)
    torques = np.zeros((N, 6))
    for t_idx in range(N):
        torques[t_idx] = compute_torques(
            traj['q'][t_idx], traj['qdot'][t_idx], traj['qddot'][t_idx], params)

    violations = []

    # Position
    q_viol = np.any((traj['q'] < q_min) | (traj['q'] > q_max))
    if q_viol:
        worst = np.max(np.maximum(q_min - traj['q'], traj['q'] - q_max))
        violations.append(f"Joint position limit exceeded by {worst:.4f} rad")
    pos_margin = float(np.min(np.minimum(traj['q'] - q_min, q_max - traj['q'])))

    # Velocity
    v_viol = np.any(np.abs(traj['qdot']) > v_max)
    if v_viol:
        worst = np.max(np.abs(traj['qdot']) - v_max)
        violations.append(f"Velocity limit exceeded by {worst:.4f} rad/s")
    vel_margin = float(np.min(v_max - np.abs(traj['qdot'])))

    # Torque
    tau_viol = np.any(np.abs(torques) > tau_max)
    if tau_viol:
        worst = np.max(np.abs(torques) - tau_max)
        violations.append(f"Torque limit exceeded by {worst:.4f} Nm")
    tau_margin = float(np.min(tau_max - np.abs(torques)))

    return {
        'satisfied':  len(violations) == 0,
        'violations': violations,
        'margin': {
            'position': pos_margin,
            'velocity': vel_margin,
            'torque':   tau_margin,
        },
    }
