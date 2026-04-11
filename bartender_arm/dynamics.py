"""
dynamics.py — Full rigid-body dynamics for the Kinova Gen3 Lite 6-DOF arm
using the Recursive Newton-Euler (RNE) algorithm.

No ROS2 dependencies. Pure numpy.

Implements:
    τ = M(q)q̈ + C(q, q̇)q̇ + G(q)

via the RNE algorithm, which is efficient for 6-DOF serial chains and
computes the same result as the full Lagrangian formulation.

Frame convention
----------------
Each link i has a body frame whose origin is the joint i axis.
The rotation R[i] expresses frame i in frame (i-1):
    R[i] = R_rpy[i] @ Rz(q[i])
where R_rpy[i] is the URDF fixed rotation for joint i.

Inertia tensors are given in each link's own body frame (as in the URDF),
with origin at the link's CoM (i.e., the URDF <inertial> frame, which has
rpy="0 0 0" for all Gen3 Lite links — no tensor rotation needed).

Gravity
-------
Gravity is handled via the D'Alembert method: initialize the base linear
acceleration as a_0 = -g = [0, 0, +9.81] m/s².  This naturally produces
gravitational torques in the backward pass without special treatment.

Usage
-----
    import yaml
    from bartender_arm.dynamics import compute_torques, gravity_torques

    with open('config/robot_params.yaml') as f:
        params = yaml.safe_load(f)

    q     = np.zeros(6)
    qdot  = np.zeros(6)
    qddot = np.zeros(6)

    tau = compute_torques(q, qdot, qddot, params)
    G   = gravity_torques(q, params)
"""

import numpy as np
from bartender_arm.kinematics import rpy_to_rotation_matrix


# ---------------------------------------------------------------------------
# Inertia tensor helpers
# ---------------------------------------------------------------------------

def _inertia_tensor(inertia_dict: dict) -> np.ndarray:
    """
    Build a 3x3 symmetric inertia tensor from a dict with keys
    ixx, ixy, ixz, iyy, iyz, izz.
    """
    ixx = inertia_dict['ixx']
    ixy = inertia_dict['ixy']
    ixz = inertia_dict['ixz']
    iyy = inertia_dict['iyy']
    iyz = inertia_dict['iyz']
    izz = inertia_dict['izz']
    return np.array([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz],
    ], dtype=float)


def _load_link_inertials(params: dict):
    """
    Extract per-link inertial data from params in joint order.

    Returns
    -------
    masses    : list of 6 floats
    coms      : list of 6 np.ndarray (3,)  — CoM in link frame
    inertias  : list of 6 np.ndarray (3,3) — inertia tensor in link frame
    """
    link_names = ['base_link', 'shoulder_link', 'arm_link',
                  'forearm_link', 'lower_wrist_link', 'upper_wrist_link']
    inertial_data = params['robot']['inertial']

    masses   = []
    coms     = []
    inertias = []
    for name in link_names:
        d = inertial_data[name]
        masses.append(float(d['mass']))
        coms.append(np.array(d['com'], dtype=float))
        inertias.append(_inertia_tensor(d['inertia']))

    return masses, coms, inertias


def _compute_link_rotations(q, params: dict):
    """
    Compute R[i] = rotation of link i frame expressed in link (i-1) frame.

    R[i] = R_rpy[i] @ Rz(q[i])

    Returns list of 6 np.ndarray, each (3,3).
    """
    joint_transforms = params['robot']['joint_transforms']
    rotations = []
    for i in range(6):
        R_rpy = rpy_to_rotation_matrix(joint_transforms[i]['rpy'])
        theta  = float(q[i])
        c, s   = np.cos(theta), np.sin(theta)
        Rz = np.array([[c, -s, 0],
                       [s,  c, 0],
                       [0,  0, 1]], dtype=float)
        rotations.append(R_rpy @ Rz)
    return rotations


def _get_joint_origins(params: dict):
    """
    Return the xyz translation of each joint in its parent frame.
    p[i] = joint_transforms[i]['xyz']  (expressed in parent frame)
    """
    return [np.array(jt['xyz'], dtype=float)
            for jt in params['robot']['joint_transforms']]


# ---------------------------------------------------------------------------
# Recursive Newton-Euler algorithm
# ---------------------------------------------------------------------------

_Z = np.array([0.0, 0.0, 1.0])  # joint axis in local frame (always Z)


def rne(q, qdot, qddot, params: dict,
        gravity=np.array([0.0, 0.0, -9.81])) -> np.ndarray:
    """
    Recursive Newton-Euler algorithm. Returns joint torque vector τ (6,).

    Parameters
    ----------
    q      : array-like (6,) — joint positions (rad)
    qdot   : array-like (6,) — joint velocities (rad/s)
    qddot  : array-like (6,) — joint accelerations (rad/s²)
    params : dict — loaded from robot_params.yaml
    gravity : array-like (3,) — gravity vector in base frame (default -Z)

    Returns
    -------
    tau : np.ndarray (6,) — joint torques (Nm)

    Algorithm
    ---------
    Forward pass (base → tip): propagate angular velocity, angular
    acceleration, and linear acceleration of each link CoM.

    Backward pass (tip → base): accumulate forces and moments, project
    onto joint axis to get torque.
    """
    q     = np.asarray(q,     dtype=float)
    qdot  = np.asarray(qdot,  dtype=float)
    qddot = np.asarray(qddot, dtype=float)
    g     = np.asarray(gravity, dtype=float)

    n = 6
    R   = _compute_link_rotations(q, params)       # R[i]: frame i in frame i-1
    p   = _get_joint_origins(params)               # p[i]: joint i origin in parent
    masses, coms, inertias = _load_link_inertials(params)

    # -----------------------------------------------------------------------
    # Forward pass — all quantities in each link's local frame
    # -----------------------------------------------------------------------
    omega   = [None] * n   # angular velocity of link i
    alpha   = [None] * n   # angular acceleration of link i
    a_joint = [None] * n   # linear acceleration of joint i origin
    a_com   = [None] * n   # linear acceleration of CoM of link i

    # D'Alembert: initialize base with negated gravity so it appears as
    # an upward "pseudo-acceleration" on the base link.
    omega_prev = np.zeros(3)
    alpha_prev = np.zeros(3)
    a_prev     = -g          # a_0 = -g (in base/world frame)

    for i in range(n):
        Ri = R[i]   # rotation of frame i in frame i-1
        RiT = Ri.T  # rotation of frame i-1 in frame i  (= R[i]^T)

        qi_dot  = qdot[i]
        qi_ddot = qddot[i]

        # Angular velocity: ω_i = R_i^T * ω_{i-1} + q̇_i * z
        omega_i = RiT @ omega_prev + qi_dot * _Z
        omega[i] = omega_i

        # Angular acceleration: α_i = R_i^T * α_{i-1} + q̈_i * z + q̇_i * (R_i^T * ω_{i-1}) × z
        alpha_i = (RiT @ alpha_prev
                   + qi_ddot * _Z
                   + qi_dot * np.cross(RiT @ omega_prev, _Z))
        alpha[i] = alpha_i

        # p_i expressed in frame i (rotate the parent-frame vector into child)
        p_i_in_child = RiT @ p[i]

        # Linear acceleration of joint i origin (expressed in frame i)
        a_i = (RiT @ a_prev
               + np.cross(alpha_i, p_i_in_child)
               + np.cross(omega_i, np.cross(omega_i, p_i_in_child)))
        a_joint[i] = a_i

        # Linear acceleration of CoM of link i
        r_c = coms[i]
        a_com_i = (a_i
                   + np.cross(alpha_i, r_c)
                   + np.cross(omega_i, np.cross(omega_i, r_c)))
        a_com[i] = a_com_i

        # Propagate to next link
        omega_prev = omega_i
        alpha_prev = alpha_i
        a_prev     = a_i

    # -----------------------------------------------------------------------
    # Backward pass — propagate forces and moments from tip to base
    # -----------------------------------------------------------------------
    tau = np.zeros(n)

    f_next = np.zeros(3)   # force exerted on link i+1 by link i (initially 0 at tip)
    n_next = np.zeros(3)   # moment at joint i+1

    for i in range(n - 1, -1, -1):
        m_i  = masses[i]
        I_i  = inertias[i]
        r_c  = coms[i]

        # Inertial force and moment at CoM
        F_i = m_i * a_com[i]
        N_i = I_i @ alpha[i] + np.cross(omega[i], I_i @ omega[i])

        # Rotation from link (i+1) to link i
        if i < n - 1:
            R_next = R[i + 1]           # R[i+1]: frame i+1 in frame i
            f_i1_in_i = R_next @ f_next  # force from link i+1 on link i+1's cut
            n_i1_in_i = R_next @ n_next

            # p_{i+1} expressed in frame i (joint i+1 origin in frame i)
            p_next_in_i = p[i + 1]   # already in parent (frame i) coords
        else:
            f_i1_in_i = np.zeros(3)
            n_i1_in_i = np.zeros(3)
            p_next_in_i = np.zeros(3)

        # Net force on link i (Newton's 2nd law)
        f_i = F_i + f_i1_in_i

        # Net moment about joint i origin
        n_i = (N_i
               + n_i1_in_i
               + np.cross(r_c, F_i)
               + np.cross(p_next_in_i, f_i1_in_i))

        # Joint torque = projection of moment onto joint axis
        tau[i] = np.dot(n_i, _Z)

        # Pass to next iteration
        f_next = f_i
        n_next = n_i

    return tau


# ---------------------------------------------------------------------------
# Higher-level dynamics functions (all derived from rne)
# ---------------------------------------------------------------------------

def compute_torques(q, qdot, qddot, params: dict) -> np.ndarray:
    """
    Full inverse dynamics: τ = M(q)q̈ + C(q,q̇)q̇ + G(q).

    Convenience wrapper around rne() with default gravity.
    """
    return rne(q, qdot, qddot, params)


def gravity_torques(q, params: dict) -> np.ndarray:
    """
    Gravitational torque vector G(q).

    G(q) = rne(q, 0, 0, params)  (qdot=0, qddot=0)
    """
    return rne(q, np.zeros(6), np.zeros(6), params)


def coriolis_vector(q, qdot, params: dict) -> np.ndarray:
    """
    Coriolis + centripetal torque vector C(q,q̇)q̇.

    = rne(q, qdot, 0, params, gravity=0)
    (zero gravity to isolate velocity-dependent terms)
    """
    return rne(q, qdot, np.zeros(6), params, gravity=np.zeros(3))


def mass_matrix(q, params: dict) -> np.ndarray:
    """
    Compute the 6x6 joint-space mass matrix M(q).

    Uses the unit-vector method:
        M[:, i] = rne(q, 0, e_i, params, gravity=0)
    where e_i is the i-th standard basis vector.

    Returns
    -------
    M : np.ndarray (6, 6) — symmetric, positive-definite
    """
    M = np.zeros((6, 6))
    for i in range(6):
        e_i = np.zeros(6)
        e_i[i] = 1.0
        M[:, i] = rne(q, np.zeros(6), e_i, params, gravity=np.zeros(3))
    return M


def kinetic_energy(q, qdot, params: dict) -> float:
    """
    Compute joint-space kinetic energy: T = 0.5 * qdot^T @ M(q) @ qdot.
    Useful for verifying M(q) is correct.
    """
    M = mass_matrix(q, params)
    qdot = np.asarray(qdot, dtype=float)
    return 0.5 * float(qdot @ M @ qdot)


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

def verify_dynamics_consistency(q, qdot, qddot, params: dict) -> float:
    """
    Verify that M(q)q̈ + C(q,q̇)q̇ + G(q) == rne(q, q̇, q̈).

    Returns the max absolute difference (should be < 1e-10).
    """
    tau_rne = rne(q, qdot, qddot, params)

    M = mass_matrix(q, params)
    C = coriolis_vector(q, qdot, params)
    G = gravity_torques(q, params)
    tau_decomp = M @ np.asarray(qddot) + C + G

    return float(np.max(np.abs(tau_rne - tau_decomp)))
