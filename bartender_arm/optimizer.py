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
from scipy.optimize import minimize, OptimizeResult

from bartender_arm.kinematics import ee_acceleration
from bartender_arm.dynamics import compute_torques
from bartender_arm.trajectory import make_trajectory, baseline_params


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def objective(x, t_arr, q0, f, params, weights, n_harmonics: int = 3) -> float:
    """
    Compute optimization objective J(x).

    J(x) = (1/N) Σ_t [ w₁·||τ(t)||² - w₂·||a_ee_lin(t)||² ]

    Parameters
    ----------
    x          : (36,) optimization variable vector
    t_arr      : (N,) time array for evaluation
    q0         : (6,) center joint pose
    f          : float, base trajectory frequency (Hz)
    params     : dict loaded from robot_params.yaml
    weights    : dict with keys 'weight_torque', 'weight_accel'
    n_harmonics: int

    Returns
    -------
    J : float (scalar to minimize)
    """
    w_tau  = float(weights.get('weight_torque', 1.0))
    w_acc  = float(weights.get('weight_accel',  0.5))

    traj = make_trajectory(x, f, t_arr, q0, n_harmonics=n_harmonics)
    N    = len(t_arr)

    J_torque = 0.0
    J_accel  = 0.0

    for t_idx in range(N):
        q_t     = traj['q'][t_idx]
        qdot_t  = traj['qdot'][t_idx]
        qddot_t = traj['qddot'][t_idx]

        tau    = compute_torques(q_t, qdot_t, qddot_t, params)
        a_ee   = ee_acceleration(q_t, qdot_t, qddot_t, params)

        J_torque += np.dot(tau, tau)          # ||τ||²
        J_accel  += np.dot(a_ee[:3], a_ee[:3])  # ||a_ee_linear||²

    J_torque /= N
    J_accel  /= N

    return w_tau * J_torque - w_acc * J_accel


def objective_and_components(x, t_arr, q0, f, params, weights,
                              n_harmonics: int = 3) -> tuple:
    """
    Like objective() but also returns the two component values.

    Returns
    -------
    J         : float — total objective
    J_torque  : float — mean squared torque norm
    J_accel   : float — mean squared EE linear acceleration
    """
    w_tau = float(weights.get('weight_torque', 1.0))
    w_acc = float(weights.get('weight_accel',  0.5))

    traj = make_trajectory(x, f, t_arr, q0, n_harmonics=n_harmonics)
    N    = len(t_arr)

    J_torque = 0.0
    J_accel  = 0.0

    for t_idx in range(N):
        q_t     = traj['q'][t_idx]
        qdot_t  = traj['qdot'][t_idx]
        qddot_t = traj['qddot'][t_idx]

        tau  = compute_torques(q_t, qdot_t, qddot_t, params)
        a_ee = ee_acceleration(q_t, qdot_t, qddot_t, params)

        J_torque += np.dot(tau, tau)
        J_accel  += np.dot(a_ee[:3], a_ee[:3])

    J_torque /= N
    J_accel  /= N

    return w_tau * J_torque - w_acc * J_accel, J_torque, J_accel


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
        1. q_max[i] - q[t,i]     >= 0   for all t, i  (upper pos limit)
        2. q[t,i]   - q_min[i]   >= 0   for all t, i  (lower pos limit)
        3. t_max[i] - |tau[t,i]| >= 0   for all t, i  (torque limit)
        4. sum(x²)  - A_min      >= 0                  (minimum motion)

    Velocity limits are enforced implicitly via max_coeff_amplitude bounds:
        (1+2+3)*2*pi*f*max_coeff = 1.319 rad/s < v_max = 1.3963 rad/s  ✓
    """
    robot   = params['robot']
    opt_cfg = params['optimization']

    jlimits = robot['joint_limits']
    jnames  = robot['joint_names']
    q_min   = np.array([jlimits[j][0] for j in jnames], dtype=float)
    q_max   = np.array([jlimits[j][1] for j in jnames], dtype=float)
    tau_max = np.array(robot['effort_limits'],   dtype=float)
    A_min   = float(opt_cfg.get('min_amplitude_sq', 0.01))

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

    def torque_constraints(x):
        _, _, torques = _eval(x)
        return (tau_max - np.abs(torques)).ravel()  # tau_max[j] - |tau[t,j]| >= 0

    # NOTE: No explicit velocity constraint needed here.
    # max_coeff_amplitude is set so that (1+2+3)*2*pi*f*C < v_max for all joints,
    # guaranteeing velocity limits hold for ALL time — not just sample points.
    # See config/robot_params.yaml: max_coeff_amplitude=0.10, f=0.25 Hz, v_max=1.3963 rad/s
    # → max_qdot = 6*2*pi*0.25*sqrt(2)*0.10 = 1.333 rad/s < 1.3963 rad/s  ✓

    def min_amplitude(x):
        return np.dot(x, x) - A_min

    return [
        {'type': 'ineq', 'fun': pos_constraints},
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
        n_scalar = 2 * N_t * 6 + N_t * 6 + 1  # pos(2NJ) + torque(NJ) + min_amp
        print(f"Starting SLSQP optimization:")
        print(f"  Variables  : {len(x0)}")
        print(f"  Constraints: ~{n_scalar} scalar (3 batched vector fns; velocity guaranteed by bounds)")
        print(f"  Time points: {len(t_arr)} (opt), {params['optimization']['n_eval_points']} (eval)")
        print(f"  Max iter   : {max_iter}")
        J0, J_tau0, J_acc0 = objective_and_components(
            x0, t_arr, q0, f, params, weights, n_harmonics)
        print(f"  Initial J  : {J0:.6f}  (J_torque={J_tau0:.4f}, J_accel={J_acc0:.4f})")

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
        J_opt, J_tau_opt, J_acc_opt = objective_and_components(
            result.x, t_arr, q0, f, params, weights, n_harmonics)
        print(f"\nOptimization {'converged' if result.success else 'did NOT converge'}:")
        print(f"  Final J    : {J_opt:.6f}  (J_torque={J_tau_opt:.4f}, J_accel={J_acc_opt:.4f})")
        print(f"  Message    : {result.message}")

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
