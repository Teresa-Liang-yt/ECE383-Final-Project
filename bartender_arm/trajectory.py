"""
trajectory.py — Fourier-series trajectory parameterization for mixing motion.

No ROS2 dependencies. Pure numpy.

The mixing trajectory is parameterized as a sum of sinusoids per joint:

    q_i(t) = q0_i + Σ_{k=1}^{K} [ a_{i,k} * sin(k·ω·t)
                                  + b_{i,k} * cos(k·ω·t) ]

where ω = 2π·f (base angular frequency).

This parameterization:
  - Is automatically periodic with period T = 1/f
  - Has analytic first and second derivatives (no finite differences needed
    during optimization, which greatly improves convergence speed)
  - Has 2·n_joints·K = 36 optimization variables (K=3, n=6)

Optimization variable layout
-----------------------------
x ∈ R^(2·6·K), organized as:

    x[  0 ..  K-1] = a[joint_0, harmonics 1..K]  (sin coeffs, joint 1)
    x[  K .. 2K-1] = a[joint_1, harmonics 1..K]
    ...
    x[5K .. 6K-1] = a[joint_5, harmonics 1..K]
    x[6K .. 7K-1] = b[joint_0, harmonics 1..K]  (cos coeffs, joint 1)
    ...
    x[11K..12K-1] = b[joint_5, harmonics 1..K]

Helper: x_to_ab(x, K) → a (6,K), b (6,K)

Baseline trajectory
-------------------
The naive (non-optimized) baseline is a simple sinusoidal oscillation
of joint_2 (shoulder) only at the base frequency:
    a[1, 0] = 0.3 rad, all other coefficients = 0.
This represents uninformed "just move the arm" mixing.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Parameter vector helpers
# ---------------------------------------------------------------------------

def n_params(n_joints: int = 6, n_harmonics: int = 3) -> int:
    """Total number of Fourier optimization variables."""
    return 2 * n_joints * n_harmonics


def x_to_ab(x, n_joints: int = 6, n_harmonics: int = 3):
    """
    Unpack flat optimization vector x into coefficient arrays.

    Returns
    -------
    a : np.ndarray (n_joints, n_harmonics) — sin coefficients
    b : np.ndarray (n_joints, n_harmonics) — cos coefficients
    """
    x = np.asarray(x, dtype=float)
    K = n_harmonics
    n = n_joints
    a = x[:n * K].reshape(n, K)
    b = x[n * K:].reshape(n, K)
    return a, b


def ab_to_x(a, b) -> np.ndarray:
    """Pack (a, b) coefficient arrays back into flat vector x."""
    return np.concatenate([a.ravel(), b.ravel()])


def baseline_params(n_joints: int = 6, n_harmonics: int = 3) -> np.ndarray:
    """
    Return the baseline (non-optimized) parameter vector.

    Baseline: only joint_2 (index 1) oscillates sinusoidally at the
    fundamental frequency with amplitude 0.3 rad.  All other coefficients
    are zero.  This is a naive mixing motion — just waving one joint.
    """
    # Peak velocity for harmonic k=1 with amplitude A: A * 2π * f
    # With f=1 Hz and velocity limit 1.396 rad/s: A_max = 1.396 / (2π) ≈ 0.222 rad
    # Use 0.18 rad to stay comfortably within limits with margin for numerical noise.
    a = np.zeros((n_joints, n_harmonics))
    b = np.zeros((n_joints, n_harmonics))
    a[1, 0] = 0.18   # joint_2, harmonic k=1, sin coefficient (~0.18 * 2π ≈ 1.13 rad/s)
    return ab_to_x(a, b)


# ---------------------------------------------------------------------------
# Trajectory evaluation
# ---------------------------------------------------------------------------

def make_trajectory(x, f: float, t_arr, q0,
                    n_joints: int = 6, n_harmonics: int = 3) -> dict:
    """
    Evaluate trajectory at time array t_arr from parameter vector x.

    Parameters
    ----------
    x          : array-like (2*n_joints*n_harmonics,) — optimization variables
    f          : float — base frequency (Hz)
    t_arr      : array-like (N,) — time points (s)
    q0         : array-like (n_joints,) — center/DC joint pose (rad)
    n_joints   : int
    n_harmonics: int

    Returns
    -------
    dict with keys:
        'q'     : (N, n_joints) joint positions
        'qdot'  : (N, n_joints) joint velocities (analytic)
        'qddot' : (N, n_joints) joint accelerations (analytic)
        't'     : (N,) time array (same as input)
    """
    x    = np.asarray(x,    dtype=float)
    t    = np.asarray(t_arr, dtype=float)
    q0   = np.asarray(q0,   dtype=float)
    a, b = x_to_ab(x, n_joints, n_harmonics)

    omega = 2.0 * np.pi * f          # base angular frequency (rad/s)
    N = len(t)

    q     = np.zeros((N, n_joints))
    qdot  = np.zeros((N, n_joints))
    qddot = np.zeros((N, n_joints))

    for i in range(n_joints):
        q[:, i]     = q0[i]
        qdot[:, i]  = 0.0
        qddot[:, i] = 0.0

        for k_idx in range(n_harmonics):
            k   = k_idx + 1            # harmonic number (1-indexed)
            kw  = k * omega            # k * ω
            kw2 = kw * kw              # (k * ω)^2

            sin_kt = np.sin(kw * t)
            cos_kt = np.cos(kw * t)

            a_ik = a[i, k_idx]
            b_ik = b[i, k_idx]

            # Position:     a*sin(kωt) + b*cos(kωt)
            q[:, i]     += a_ik * sin_kt + b_ik * cos_kt

            # Velocity:     a*kω*cos(kωt) - b*kω*sin(kωt)
            qdot[:, i]  += a_ik * kw * cos_kt - b_ik * kw * sin_kt

            # Acceleration: -a*(kω)²*sin(kωt) - b*(kω)²*cos(kωt)
            qddot[:, i] += -a_ik * kw2 * sin_kt - b_ik * kw2 * cos_kt

    return {'q': q, 'qdot': qdot, 'qddot': qddot, 't': t}


def sample_trajectory(traj: dict, n_waypoints: int) -> dict:
    """
    Downsample trajectory to n_waypoints for ROS JointTrajectory messages.

    Uses uniform spacing; if n_waypoints >= original length, returns as-is.
    """
    N = len(traj['t'])
    if n_waypoints >= N:
        return traj
    indices = np.round(np.linspace(0, N - 1, n_waypoints)).astype(int)
    return {
        'q':     traj['q'][indices],
        'qdot':  traj['qdot'][indices],
        'qddot': traj['qddot'][indices],
        't':     traj['t'][indices],
    }


# ---------------------------------------------------------------------------
# Trajectory metrics
# ---------------------------------------------------------------------------

def trajectory_metrics(traj: dict, torques: np.ndarray,
                       ee_accels: np.ndarray) -> dict:
    """
    Compute summary metrics for a trajectory.

    Parameters
    ----------
    traj     : dict from make_trajectory()
    torques  : (N, 6) array of joint torques
    ee_accels: (N, 3) array of EE linear accelerations

    Returns
    -------
    dict with:
        peak_torque       : max |τ| across all joints and time
        rms_torque        : RMS of torque norm across time
        mean_torque_norm  : mean of ||τ(t)||₂ across time
        peak_ee_accel     : max ||a_ee(t)||₂
        rms_ee_accel      : RMS of ||a_ee(t)||₂
        mean_ee_accel     : mean of ||a_ee(t)||₂
        peak_velocity     : max |q̇| across all joints and time
        rms_velocity      : RMS of velocity norm across time
    """
    tau_norm   = np.linalg.norm(torques,   axis=1)   # (N,)
    accel_norm = np.linalg.norm(ee_accels, axis=1)   # (N,)
    qdot       = traj['qdot']

    return {
        'peak_torque':      float(np.max(np.abs(torques))),
        'rms_torque':       float(np.sqrt(np.mean(tau_norm ** 2))),
        'mean_torque_norm': float(np.mean(tau_norm)),
        'peak_ee_accel':    float(np.max(accel_norm)),
        'rms_ee_accel':     float(np.sqrt(np.mean(accel_norm ** 2))),
        'mean_ee_accel':    float(np.mean(accel_norm)),
        'peak_velocity':    float(np.max(np.abs(qdot))),
        'rms_velocity':     float(np.sqrt(np.mean(np.linalg.norm(qdot, axis=1) ** 2))),
    }


def print_metrics_table(baseline_metrics: dict, optimized_metrics: dict):
    """Print a formatted comparison table to stdout."""
    keys = [
        ('peak_torque',       'Peak torque (Nm)'),
        ('rms_torque',        'RMS torque (Nm)'),
        ('mean_torque_norm',  'Mean torque norm (Nm)'),
        ('peak_ee_accel',     'Peak EE accel (m/s²)'),
        ('rms_ee_accel',      'RMS EE accel (m/s²)'),
        ('mean_ee_accel',     'Mean EE accel (m/s²)'),
        ('peak_velocity',     'Peak joint vel (rad/s)'),
    ]
    header = f"{'Metric':<28} {'Baseline':>12} {'Optimized':>12} {'Change':>10}"
    print('\n' + '=' * len(header))
    print(header)
    print('=' * len(header))
    for key, label in keys:
        b = baseline_metrics[key]
        o = optimized_metrics[key]
        pct = (o - b) / (abs(b) + 1e-12) * 100.0
        sign = '+' if pct >= 0 else ''
        print(f"{label:<28} {b:>12.4f} {o:>12.4f} {sign}{pct:>8.1f}%")
    print('=' * len(header) + '\n')


# ---------------------------------------------------------------------------
# ROS2 message conversion (imported conditionally to avoid hard ROS2 dep)
# ---------------------------------------------------------------------------

def trajectory_to_ros_msg(traj: dict, joint_names: list, start_stamp=None):
    """
    Convert trajectory dict to a trajectory_msgs/JointTrajectory ROS2 message.

    This function imports ROS2 types at call time so the rest of the module
    remains usable without a ROS2 installation.

    Parameters
    ----------
    traj        : dict from make_trajectory() or sample_trajectory()
    joint_names : list of str — must match controller's expected names
    start_stamp : rclpy.time.Time or None — if None, uses header.stamp=0

    Returns
    -------
    trajectory_msgs.msg.JointTrajectory
    """
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # type: ignore
    from builtin_interfaces.msg import Duration  # type: ignore

    msg = JointTrajectory()
    msg.joint_names = list(joint_names)

    if start_stamp is not None:
        msg.header.stamp = start_stamp.to_msg()

    q     = traj['q']
    qdot  = traj['qdot']
    t_arr = traj['t']

    for idx in range(len(t_arr)):
        pt = JointTrajectoryPoint()
        pt.positions  = q[idx].tolist()
        pt.velocities = qdot[idx].tolist()

        t_sec    = float(t_arr[idx])
        sec_int  = int(t_sec)
        nanosec  = int(round((t_sec - sec_int) * 1e9))
        pt.time_from_start = Duration(sec=sec_int, nanosec=nanosec)

        msg.points.append(pt)

    return msg
