"""
kinematics.py — Forward kinematics, geometric Jacobian, and end-effector
acceleration for the Kinova Gen3 Lite 6-DOF arm.

No ROS2 dependencies. Pure numpy.

Convention (matches URDF / ros2_kortex gen3_lite.urdf):
  - Each joint i has a fixed transform T_fixed[i] from parent frame to
    child frame (xyz offset + rpy rotation), followed by a rotation
    about the child's LOCAL Z-axis by joint angle q[i].
  - URDF rpy = Rx(roll) * Ry(pitch) * Rz(yaw)  (intrinsic XYZ order).
  - All 6 joints are revolute with axis = [0, 0, 1] in their own frame.

Usage:
    import yaml
    from bartender_arm.kinematics import forward_kinematics, geometric_jacobian

    with open('config/robot_params.yaml') as f:
        params = yaml.safe_load(f)

    q = np.zeros(6)
    T_links, T_ee = forward_kinematics(q, params)
    J = geometric_jacobian(q, params)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Primitive rotation / transform helpers
# ---------------------------------------------------------------------------

def rx(angle: float) -> np.ndarray:
    """4x4 homogeneous rotation about X-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1,  0,  0, 0],
        [0,  c, -s, 0],
        [0,  s,  c, 0],
        [0,  0,  0, 1],
    ], dtype=float)


def ry(angle: float) -> np.ndarray:
    """4x4 homogeneous rotation about Y-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ], dtype=float)


def rz(angle: float) -> np.ndarray:
    """4x4 homogeneous rotation about Z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=float)


def rpy_to_rotation_matrix(rpy) -> np.ndarray:
    """
    Convert URDF rpy (roll, pitch, yaw) to a 3x3 rotation matrix.

    URDF convention: R = Rx(roll) * Ry(pitch) * Rz(yaw)
    """
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    # R = Rx @ Ry @ Rz
    R = np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [  -sp,          cp*sr,             cp*cr   ],
    ], dtype=float)
    return R


def transform_from_xyz_rpy(xyz, rpy) -> np.ndarray:
    """
    Build a 4x4 homogeneous transform from a URDF xyz+rpy pair.

    T = [R | p]
        [0 | 1]
    where R = rpy_to_rotation_matrix(rpy), p = xyz.
    """
    T = np.eye(4)
    T[:3, :3] = rpy_to_rotation_matrix(rpy)
    T[:3,  3] = xyz
    return T


def rot_z_matrix(theta: float) -> np.ndarray:
    """3x3 rotation matrix about Z by theta."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)


def skew(v) -> np.ndarray:
    """3x3 skew-symmetric (cross-product) matrix for vector v."""
    return np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0],
    ], dtype=float)


# ---------------------------------------------------------------------------
# Core kinematics
# ---------------------------------------------------------------------------

def _precompute_fixed_transforms(params: dict):
    """
    Pre-compute the list of fixed 4x4 transforms T_fixed[i] from URDF.
    Returns a list of 6 numpy arrays.
    """
    transforms = []
    for jt in params['robot']['joint_transforms']:
        xyz = jt['xyz']
        rpy = jt['rpy']
        transforms.append(transform_from_xyz_rpy(xyz, rpy))
    return transforms


def forward_kinematics(q, params: dict):
    """
    Compute forward kinematics for the 6-DOF Gen3 Lite.

    Parameters
    ----------
    q : array-like, shape (6,)
        Joint angles in radians.
    params : dict
        Loaded from robot_params.yaml.

    Returns
    -------
    T_links : list of 7 np.ndarray, each (4,4)
        T_links[0] = identity (world/base frame).
        T_links[i] = accumulated transform to joint i's frame AFTER
                     applying q[i-1], for i = 1..6.
    T_ee : np.ndarray, shape (4,4)
        End-effector transform in world frame (same as T_links[6] unless
        a non-trivial ee_offset is configured).
    """
    q = np.asarray(q, dtype=float)
    fixed_transforms = _precompute_fixed_transforms(params)

    T = np.eye(4)
    T_links = [T.copy()]  # T_links[0] = base = identity

    for i in range(6):
        T_fixed = fixed_transforms[i]
        T_joint = rz(q[i])
        T = T @ T_fixed @ T_joint
        T_links.append(T.copy())

    # Apply end-effector offset if any
    ee = params['robot'].get('ee_offset', {'xyz': [0, 0, 0], 'rpy': [0, 0, 0]})
    T_ee_offset = transform_from_xyz_rpy(ee['xyz'], ee['rpy'])
    T_ee = T @ T_ee_offset

    return T_links, T_ee


def geometric_jacobian(q, params: dict) -> np.ndarray:
    """
    Compute the 6x6 geometric Jacobian for the end-effector.

    J = [J_v]   where J_v[:,i] = z_i × (p_ee - p_i)   (linear velocity)
        [J_w]         J_w[:,i] = z_i                    (angular velocity)

    z_i is the joint i rotation axis expressed in world frame:
        z_i = T_links[i][:3, :3] @ [0, 0, 1]

    p_i is the origin of joint i frame in world: T_links[i][:3, 3]
    p_ee is the end-effector position: T_ee[:3, 3]

    Parameters
    ----------
    q : array-like, shape (6,)
    params : dict

    Returns
    -------
    J : np.ndarray, shape (6, 6)
        Rows 0-2: linear velocity components.
        Rows 3-5: angular velocity components.
    """
    q = np.asarray(q, dtype=float)
    T_links, T_ee = forward_kinematics(q, params)
    p_ee = T_ee[:3, 3]

    J = np.zeros((6, 6))
    for i in range(6):
        # Joint i's axis and origin are in the frame AFTER the fixed transform
        # T_fixed[i] but BEFORE the joint rotation Rz(q[i]).
        # Since Rz(q) @ [0,0,1] = [0,0,1], the z-axis of T_links[i+1] equals
        # the z-axis of (T_links[i] @ T_fixed[i]).  The translation is identical
        # because Rz introduces no offset.  So T_links[i+1] gives both correctly.
        z_i = T_links[i + 1][:3, 2]   # 3rd column of rotation = z-axis in world
        p_i = T_links[i + 1][:3, 3]   # joint origin in world frame
        J[:3, i] = np.cross(z_i, p_ee - p_i)  # linear
        J[3:, i] = z_i                          # angular

    return J


def ee_velocity(q, qdot, params: dict) -> np.ndarray:
    """
    Compute end-effector velocity (6-vector: [vx, vy, vz, wx, wy, wz]).

    v_ee = J(q) @ qdot
    """
    J = geometric_jacobian(q, params)
    return J @ np.asarray(qdot, dtype=float)


def _jacobian_dot(q, qdot, params: dict, dt: float = 1e-7) -> np.ndarray:
    """
    Numerically compute Jdot = dJ/dt using first-order finite difference.

    Jdot ≈ (J(q + qdot*dt) - J(q)) / dt
    """
    q = np.asarray(q, dtype=float)
    qdot = np.asarray(qdot, dtype=float)
    J0 = geometric_jacobian(q, params)
    J1 = geometric_jacobian(q + qdot * dt, params)
    return (J1 - J0) / dt


def ee_acceleration(q, qdot, qddot, params: dict) -> np.ndarray:
    """
    Compute end-effector acceleration (6-vector: [ax, ay, az, αx, αy, αz]).

    a_ee = J(q) @ qddot + Jdot(q, qdot) @ qdot

    Parameters
    ----------
    q     : array-like, shape (6,)
    qdot  : array-like, shape (6,)
    qddot : array-like, shape (6,)
    params : dict

    Returns
    -------
    a_ee : np.ndarray, shape (6,)
        Linear acceleration in indices 0:3, angular in 3:6.
    """
    q     = np.asarray(q,     dtype=float)
    qdot  = np.asarray(qdot,  dtype=float)
    qddot = np.asarray(qddot, dtype=float)

    J    = geometric_jacobian(q, params)
    Jdot = _jacobian_dot(q, qdot, params)
    return J @ qddot + Jdot @ qdot


# ---------------------------------------------------------------------------
# Utility: end-effector position only
# ---------------------------------------------------------------------------

def ee_position(q, params: dict) -> np.ndarray:
    """Return end-effector position [x, y, z] in world frame."""
    _, T_ee = forward_kinematics(q, params)
    return T_ee[:3, 3].copy()


def ee_pose(q, params: dict):
    """Return (position, rotation_matrix) of end-effector."""
    _, T_ee = forward_kinematics(q, params)
    return T_ee[:3, 3].copy(), T_ee[:3, :3].copy()


# ---------------------------------------------------------------------------
# Verification helpers (called from tests)
# ---------------------------------------------------------------------------

def verify_jacobian_numerically(q, params: dict, eps: float = 1e-6) -> float:
    """
    Check J[:,i] against finite-difference FK perturbation.

    Returns max absolute error across all columns. Should be < 1e-5.
    """
    J_analytic = geometric_jacobian(q, params)
    _, T_ee0 = forward_kinematics(q, params)
    p0 = T_ee0[:3, 3]

    max_err = 0.0
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        _, T_ee1 = forward_kinematics(q + dq, params)
        p1 = T_ee1[:3, 3]
        Jv_fd = (p1 - p0) / eps
        err = np.max(np.abs(J_analytic[:3, i] - Jv_fd))
        max_err = max(max_err, err)

    return max_err
