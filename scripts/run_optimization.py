#!/usr/bin/env python3
"""
run_optimization.py — Standalone offline optimization script.

No ROS2 required. Run directly with:
    cd ECE383-Final-Project
    python3 scripts/run_optimization.py --mode amplitude
    python3 scripts/run_optimization.py --mode frequency

Modes (defined in config/robot_params.yaml -> trajectory_modes):
    amplitude  — low frequency (0.25 Hz), large coefficients (0.10 rad) → big sweeping motion
    frequency  — high frequency (1.0 Hz), small coefficients (0.026 rad) → fast tight mixing

Output files (written to current directory or --output-dir):
    comparison_plots_<mode>.png    — 4-panel figure
    baseline_trajectory.csv        — naive baseline trajectory + torques
    optimized_trajectory_<mode>.csv
    <mode>_coeffs.npy              — optimal coefficients for ROS trajectory publisher
"""

import os
import sys
import argparse
import time

import numpy as np
import yaml

# Allow running from the repo root without installing the package
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from bartender_arm.kinematics import (
    forward_kinematics, ee_acceleration, verify_jacobian_numerically
)
from bartender_arm.dynamics import (
    compute_torques, gravity_torques, verify_dynamics_consistency
)
from bartender_arm.trajectory import (
    make_trajectory, baseline_params, trajectory_metrics,
    print_metrics_table
)
from bartender_arm.optimizer import (
    multi_start_optimization, global_optimization, check_constraints
)


# ---------------------------------------------------------------------------
# Load parameters
# ---------------------------------------------------------------------------

def load_params(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Verification checks (run before optimization)
# ---------------------------------------------------------------------------

def run_verification(params: dict, verbose: bool = True):
    """Run sanity checks on kinematics and dynamics."""
    if verbose:
        print("=" * 60)
        print("VERIFICATION CHECKS")
        print("=" * 60)

    q_home = np.zeros(6)

    # 1. FK at home pose
    _, T_ee = forward_kinematics(q_home, params)
    p_ee = T_ee[:3, 3]
    if verbose:
        print(f"FK home pose — EE position: [{p_ee[0]:.4f}, {p_ee[1]:.4f}, {p_ee[2]:.4f}] m")

    # 2. Jacobian numerical check
    jac_err = verify_jacobian_numerically(q_home, params)
    status = "PASS" if jac_err < 1e-5 else "FAIL"
    if verbose:
        print(f"Jacobian FD check — max error: {jac_err:.2e}  [{status}]")

    # 3. Gravity torques at home pose
    G = gravity_torques(q_home, params)
    if verbose:
        print(f"Gravity torques at home: {np.round(G, 4)} Nm")
        print(f"  (τ₁ should be ≈ 0, τ₂ should be largest positive)")

    # 4. Dynamics consistency
    q_test    = np.array([0.1, 0.3, -0.2, 0.1, 0.5, 0.0])
    qdot_test = np.array([0.1, 0.2, -0.1, 0.05, 0.15, 0.0])
    qddot_test = np.array([0.5, -0.3, 0.2, -0.1, 0.3, 0.0])
    dyn_err = verify_dynamics_consistency(q_test, qdot_test, qddot_test, params)
    status = "PASS" if dyn_err < 1e-8 else "FAIL"
    if verbose:
        print(f"Dynamics consistency (M*q̈+C+G vs RNE) — max error: {dyn_err:.2e}  [{status}]")
        print()


# ---------------------------------------------------------------------------
# Evaluate a trajectory and collect all metrics
# ---------------------------------------------------------------------------

def evaluate_trajectory(x, f, t_arr, q0, params, label: str = "",
                        n_harmonics: int = 3) -> dict:
    """
    Evaluate trajectory x and return metrics dict with raw arrays.
    """
    traj = make_trajectory(x, f, t_arr, q0, n_harmonics=n_harmonics)
    N = len(t_arr)

    torques   = np.zeros((N, 6))
    ee_accels = np.zeros((N, 3))
    ee_pos    = np.zeros((N, 3))

    for t_idx in range(N):
        q_t     = traj['q'][t_idx]
        qdot_t  = traj['qdot'][t_idx]
        qddot_t = traj['qddot'][t_idx]

        torques[t_idx]   = compute_torques(q_t, qdot_t, qddot_t, params)
        a_ee             = ee_acceleration(q_t, qdot_t, qddot_t, params)
        ee_accels[t_idx] = a_ee[:3]

        _, T_ee  = forward_kinematics(q_t, params)
        ee_pos[t_idx] = T_ee[:3, 3]

    metrics = trajectory_metrics(traj, torques, ee_accels)

    return {
        'traj':     traj,
        'torques':  torques,
        'ee_accels': ee_accels,
        'ee_pos':   ee_pos,
        'metrics':  metrics,
        'label':    label,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(baseline_data: dict, optimized_data: dict,
                    params: dict, output_path: str):
    """Generate 4-panel comparison figure and save to output_path."""
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for saving to file
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    t_b = baseline_data['traj']['t']
    t_o = optimized_data['traj']['t']

    colors_joints = plt.cm.tab10(np.linspace(0, 0.6, 6))

    # -----------------------------------------------------------------------
    # Panel 1: Joint torques vs time
    # -----------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    jnames = params['robot']['joint_names']
    for j in range(6):
        tau_b = baseline_data['torques'][:, j]
        tau_o = optimized_data['torques'][:, j]
        ax1.plot(t_b, tau_b, '--', color=colors_joints[j], alpha=0.7,
                 label=f'{jnames[j]} base')
        ax1.plot(t_o, tau_o, '-',  color=colors_joints[j], alpha=0.9,
                 label=f'{jnames[j]} opt')

    # Add torque limits as horizontal lines
    tau_max = params['robot']['effort_limits']
    ax1.axhline(y= max(tau_max), color='red', linestyle=':', linewidth=1.5,
                label=f'±τ_max')
    ax1.axhline(y=-max(tau_max), color='red', linestyle=':', linewidth=1.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Torque (Nm)')
    ax1.set_title('Joint Torques: Baseline (- -) vs Optimized (—)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=6, ncol=2, loc='upper right')

    # -----------------------------------------------------------------------
    # Panel 2: Joint velocities vs time
    # -----------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    for j in range(6):
        vb = baseline_data['traj']['qdot'][:, j]
        vo = optimized_data['traj']['qdot'][:, j]
        ax2.plot(t_b, vb, '--', color=colors_joints[j], alpha=0.7)
        ax2.plot(t_o, vo, '-',  color=colors_joints[j], alpha=0.9)

    v_max = max(params['robot']['velocity_limits'])
    ax2.axhline(y= v_max, color='orange', linestyle=':', linewidth=1.5,
                label='±v_max')
    ax2.axhline(y=-v_max, color='orange', linestyle=':', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (rad/s)')
    ax2.set_title('Joint Velocities: Baseline (- -) vs Optimized (—)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # -----------------------------------------------------------------------
    # Panel 3: End-effector acceleration magnitude vs time
    # -----------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    accel_b = np.linalg.norm(baseline_data['ee_accels'],  axis=1)
    accel_o = np.linalg.norm(optimized_data['ee_accels'], axis=1)
    ax3.plot(t_b, accel_b, '--', color='steelblue',  lw=2, label='Baseline')
    ax3.plot(t_o, accel_o, '-',  color='darkorange', lw=2, label='Optimized')
    ax3.fill_between(t_o, accel_o, accel_b,
                     where=(accel_o > accel_b),
                     alpha=0.2, color='green', label='Improvement')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('||a_EE|| (m/s²)')
    ax3.set_title('End-Effector Acceleration Magnitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Annotate mean values
    ax3.axhline(y=np.mean(accel_b), color='steelblue',  linestyle=':', alpha=0.7,
                label=f'mean_base={np.mean(accel_b):.2f}')
    ax3.axhline(y=np.mean(accel_o), color='darkorange', linestyle=':', alpha=0.7,
                label=f'mean_opt={np.mean(accel_o):.2f}')
    ax3.legend(fontsize=8)

    # -----------------------------------------------------------------------
    # Panel 4: End-effector path projected onto shake axis vs perpendicular
    # -----------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    pos_b = baseline_data['ee_pos']
    pos_o = optimized_data['ee_pos']

    # Project onto shake axis and one perpendicular for visual clarity.
    # Good directional trajectories appear as elongated ovals; circular ones as circles.
    shake_axis = np.array(params['optimization'].get('shake_axis', [1.0, 0.0, 0.0]), dtype=float)
    shake_axis = shake_axis / np.linalg.norm(shake_axis)
    # Pick perpendicular: world axis most orthogonal to shake_axis
    candidates = [np.array([1., 0., 0.]), np.array([0., 0., 1.]), np.array([0., 1., 0.])]
    perp_axis = min(candidates, key=lambda v: abs(np.dot(v, shake_axis)))
    perp_axis = perp_axis - np.dot(perp_axis, shake_axis) * shake_axis
    perp_axis = perp_axis / np.linalg.norm(perp_axis)

    proj_b = pos_b @ shake_axis
    perp_b = pos_b @ perp_axis
    proj_o = pos_o @ shake_axis
    perp_o = pos_o @ perp_axis

    from matplotlib.collections import LineCollection

    def colored_line(ax, x, y, c, cmap='Blues', lw=2, label=''):
        pts  = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap=cmap, linewidth=lw)
        lc.set_array(c)
        ax.add_collection(lc)
        ax.plot([], [], color=plt.cm.get_cmap(cmap)(0.7), label=label, lw=lw)

    colored_line(ax4, proj_b, perp_b,
                 np.linspace(0, 1, len(t_b)), cmap='Blues', label='Baseline')
    colored_line(ax4, proj_o, perp_o,
                 np.linspace(0, 1, len(t_o)), cmap='Oranges', label='Optimized')

    ax4.autoscale()
    shake_label = f"[{shake_axis[0]:.0f},{shake_axis[1]:.0f},{shake_axis[2]:.0f}]"
    perp_label  = f"[{perp_axis[0]:.0f},{perp_axis[1]:.0f},{perp_axis[2]:.0f}]"
    ax4.set_xlabel(f'Along shake axis {shake_label} (m)')
    ax4.set_ylabel(f'Perpendicular {perp_label} (m)')
    ax4.set_title('EE Path: shake axis (horiz) vs perpendicular (vert)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal', adjustable='datalim')

    # -----------------------------------------------------------------------
    # Title and metrics summary
    # -----------------------------------------------------------------------
    bm = baseline_data['metrics']
    om = optimized_data['metrics']
    pct_torque = (om['rms_torque'] - bm['rms_torque']) / (bm['rms_torque'] + 1e-9) * 100
    pct_accel  = (om['mean_ee_accel'] - bm['mean_ee_accel']) / (bm['mean_ee_accel'] + 1e-9) * 100

    fig.suptitle(
        f"Mixing Trajectory Optimization — Kinova Gen3 Lite\n"
        f"RMS torque: {bm['rms_torque']:.2f} → {om['rms_torque']:.2f} Nm ({pct_torque:+.1f}%)   |   "
        f"Mean EE accel: {bm['mean_ee_accel']:.2f} → {om['mean_ee_accel']:.2f} m/s² ({pct_accel:+.1f}%)",
        fontsize=11
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot → {output_path}")


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def save_trajectory_csv(traj: dict, torques: np.ndarray, path: str):
    """Save trajectory + torques to CSV."""
    header = 't,' + ','.join(f'q{i+1}' for i in range(6)) + ',' \
           + ','.join(f'qdot{i+1}' for i in range(6)) + ',' \
           + ','.join(f'qddot{i+1}' for i in range(6)) + ',' \
           + ','.join(f'tau{i+1}' for i in range(6))

    data = np.column_stack([
        traj['t'].reshape(-1, 1),
        traj['q'],
        traj['qdot'],
        traj['qddot'],
        torques,
    ])
    np.savetxt(path, data, delimiter=',', header=header, comments='')
    print(f"Saved trajectory CSV → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Offline mixing trajectory optimization for Kinova Gen3 Lite')
    parser.add_argument(
        '--mode', default='amplitude', choices=['amplitude', 'frequency'],
        help='Trajectory mode: amplitude (0.25 Hz, 0.10 rad) or frequency (1.0 Hz, 0.026 rad)')
    parser.add_argument(
        '--config', default=None,
        help='Path to robot_params.yaml (default: auto-detect from script location)')
    parser.add_argument(
        '--output-dir', default='.',
        help='Directory for output files (default: current directory)')
    parser.add_argument(
        '--skip-verification', action='store_true',
        help='Skip kinematics/dynamics sanity checks')
    parser.add_argument(
        '--no-optimize', action='store_true',
        help='Skip optimization, only evaluate baseline')
    parser.add_argument(
        '--load-coeffs', default=None,
        help='Load previously saved coefficients .npy instead of re-optimizing')
    parser.add_argument(
        '--restarts', type=int, default=None,
        help='Number of random restarts for multi-start optimization '
             '(default: read from optimization.n_restarts in config, fallback 8)')
    parser.add_argument(
        '--global', dest='use_global', action='store_true',
        help='Use Differential Evolution + SLSQP instead of multi-start SLSQP')
    args = parser.parse_args()

    # Resolve config path
    if args.config is None:
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        repo_root   = os.path.dirname(script_dir)
        config_path = os.path.join(repo_root, 'config', 'robot_params.yaml')
    else:
        config_path = args.config

    print(f"Loading params from: {config_path}")
    params = load_params(config_path)

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Resolve mode-specific trajectory parameters
    # -----------------------------------------------------------------------
    mode     = args.mode
    mode_cfg = params['trajectory_modes'][mode]
    opt_cfg  = params['optimization']

    f          = float(mode_cfg['frequency'])
    duration   = float(mode_cfg['duration'])
    n_harm     = int(mode_cfg['n_harmonics'])
    max_coeff  = float(mode_cfg['max_coeff_amplitude'])
    N_opt      = int(opt_cfg['n_opt_points'])
    N_eval     = int(opt_cfg['n_eval_points'])
    q0         = np.array(params['trajectory']['mixing_center_joints'], dtype=float)

    print(f"Mode: {mode}  |  f={f} Hz, duration={duration}s, "
          f"max_coeff={max_coeff} rad, n_harmonics={n_harm}")

    t_opt  = np.linspace(0, duration, N_opt,  endpoint=False)
    t_eval = np.linspace(0, duration, N_eval, endpoint=False)

    # -----------------------------------------------------------------------
    # Baseline trajectory
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("EVALUATING BASELINE TRAJECTORY")
    print("=" * 60)
    x_baseline = baseline_params(n_joints=6, n_harmonics=n_harm)
    t0 = time.time()
    baseline_data = evaluate_trajectory(
        x_baseline, f, t_eval, q0, params, label='Baseline', n_harmonics=n_harm)
    print(f"Evaluation time: {time.time()-t0:.2f}s")

    # Constraint check for baseline
    baseline_check = check_constraints(x_baseline, t_eval, q0, f, params, n_harm)
    print(f"Baseline constraints: {'satisfied' if baseline_check['satisfied'] else 'VIOLATED'}")
    if baseline_check['violations']:
        for v in baseline_check['violations']:
            print(f"  ! {v}")
    print()

    # -----------------------------------------------------------------------
    # Optimization
    # -----------------------------------------------------------------------
    n_restarts = args.restarts if args.restarts is not None else \
        int(params['optimization'].get('n_restarts', 8))

    if args.load_coeffs:
        print(f"Loading optimized coefficients from {args.load_coeffs}")
        x_opt = np.load(args.load_coeffs)
    elif not args.no_optimize:
        t0 = time.time()
        if args.use_global:
            print("=" * 60)
            print("RUNNING GLOBAL OPTIMIZATION (DE + SLSQP)")
            print("=" * 60)
            result = global_optimization(
                q0=q0, f=f, t_arr_opt=t_opt, params=params,
                n_harmonics=n_harm, verbose=True, max_coeff=max_coeff)
        else:
            print("=" * 60)
            print(f"RUNNING MULTI-START SLSQP OPTIMIZATION ({n_restarts} starts)")
            print("=" * 60)
            result = multi_start_optimization(
                q0=q0, f=f, t_arr_opt=t_opt, params=params,
                n_restarts=n_restarts, n_harmonics=n_harm, verbose=True,
                max_coeff=max_coeff)
        elapsed = time.time() - t0
        print(f"\nTotal optimization wall time: {elapsed:.1f}s")
        x_opt = result.x
    else:
        print("Skipping optimization (--no-optimize)")
        x_opt = x_baseline

    # -----------------------------------------------------------------------
    # Evaluate optimized trajectory
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATING OPTIMIZED TRAJECTORY")
    print("=" * 60)
    t0 = time.time()
    optimized_data = evaluate_trajectory(
        x_opt, f, t_eval, q0, params, label='Optimized', n_harmonics=n_harm)
    print(f"Evaluation time: {time.time()-t0:.2f}s")

    # Constraint check for optimized
    opt_check = check_constraints(x_opt, t_eval, q0, f, params, n_harm)
    print(f"Optimized constraints: {'satisfied' if opt_check['satisfied'] else 'VIOLATED'}")
    if opt_check['violations']:
        for v in opt_check['violations']:
            print(f"  ! {v}")
    print(f"Constraint margins — position: {opt_check['margin']['position']:.4f} rad, "
          f"velocity: {opt_check['margin']['velocity']:.4f} rad/s, "
          f"torque: {opt_check['margin']['torque']:.4f} Nm")

    # -----------------------------------------------------------------------
    # Print comparison table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print_metrics_table(baseline_data['metrics'], optimized_data['metrics'])

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    plot_path     = os.path.join(args.output_dir, f'comparison_plots_{mode}.png')
    baseline_csv  = os.path.join(args.output_dir, 'baseline_trajectory.csv')
    opt_csv       = os.path.join(args.output_dir, f'optimized_trajectory_{mode}.csv')
    coeffs_path   = os.path.join(args.output_dir, mode_cfg['coeffs_file'])

    plot_comparison(baseline_data, optimized_data, params, plot_path)
    save_trajectory_csv(baseline_data['traj'],  baseline_data['torques'],  baseline_csv)
    save_trajectory_csv(optimized_data['traj'], optimized_data['torques'], opt_csv)

    np.save(coeffs_path, x_opt)
    print(f"Saved optimized coefficients → {coeffs_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
