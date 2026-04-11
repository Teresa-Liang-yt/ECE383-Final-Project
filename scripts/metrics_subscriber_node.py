#!/usr/bin/env python3
"""
metrics_subscriber_node.py — ROS2 node that listens to /joint_states,
computes real-time joint torques and EE acceleration via the dynamics model,
and publishes the results for monitoring and logging.

Subscribes to:
    /joint_states  (sensor_msgs/msg/JointState)

Publishes:
    /bartender/joint_torques    (std_msgs/msg/Float64MultiArray)  — 6 torques (Nm)
    /bartender/ee_acceleration  (std_msgs/msg/Float64MultiArray)  — 3 EE linear accels (m/s²)
    /bartender/metrics_summary  (std_msgs/msg/Float64MultiArray)  — [peak_tau, rms_tau, mean_accel]

Also logs all data to /tmp/bartender_metrics.csv for offline analysis.

Notes
-----
- /joint_states may arrive in alphabetical order or include gripper joints.
  This node always reorders by name before processing.
- qddot is estimated via finite difference of qdot (first-order backward diff).
  The first few samples will have noisy qddot estimates — this is expected.
- Joint torques computed here are model-based estimates from inverse dynamics,
  not measured values from hardware sensors.
"""

import os
import sys
import csv
import time
from collections import deque

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_SCRIPTS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from bartender_arm.dynamics import compute_torques
from bartender_arm.kinematics import ee_acceleration


JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3',
               'joint_4', 'joint_5', 'joint_6']


def _float64_multiarray(values):
    """Pack a list/array of floats into a Float64MultiArray message."""
    msg = Float64MultiArray()
    msg.layout.dim.append(
        MultiArrayDimension(label='data', size=len(values), stride=len(values)))
    msg.data = [float(v) for v in values]
    return msg


class MetricsSubscriberNode(Node):
    def __init__(self):
        super().__init__('metrics_subscriber')

        # ---------------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------------
        self.declare_parameter('config_path', '')
        self.declare_parameter('log_path', '/tmp/bartender_metrics.csv')
        self.declare_parameter('window_size', 100)  # samples for running stats

        cfg_path    = self.get_parameter('config_path').value
        log_path    = self.get_parameter('log_path').value
        window_size = int(self.get_parameter('window_size').value)

        if not cfg_path:
            cfg_path = os.path.join(_REPO_ROOT, 'config', 'robot_params.yaml')
        with open(cfg_path) as f:
            self.params = yaml.safe_load(f)

        # ---------------------------------------------------------------
        # State for finite-difference qddot estimation
        # ---------------------------------------------------------------
        self.prev_q    = None
        self.prev_qdot = None
        self.prev_time = None

        # Running stats windows (deques)
        self.tau_window   = deque(maxlen=window_size)
        self.accel_window = deque(maxlen=window_size)

        # ---------------------------------------------------------------
        # CSV logger
        # ---------------------------------------------------------------
        self.csv_file = open(log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'ros_time',
            'q1', 'q2', 'q3', 'q4', 'q5', 'q6',
            'qdot1', 'qdot2', 'qdot3', 'qdot4', 'qdot5', 'qdot6',
            'qddot1', 'qddot2', 'qddot3', 'qddot4', 'qddot5', 'qddot6',
            'tau1', 'tau2', 'tau3', 'tau4', 'tau5', 'tau6',
            'ee_ax', 'ee_ay', 'ee_az',
        ])
        self.get_logger().info(f"Logging metrics to {log_path}")

        # ---------------------------------------------------------------
        # Subscriber and publishers
        # ---------------------------------------------------------------
        self.sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            qos_profile=10,
        )
        self.torque_pub  = self.create_publisher(
            Float64MultiArray, '/bartender/joint_torques',   10)
        self.accel_pub   = self.create_publisher(
            Float64MultiArray, '/bartender/ee_acceleration', 10)
        self.summary_pub = self.create_publisher(
            Float64MultiArray, '/bartender/metrics_summary', 10)

        self.get_logger().info("MetricsSubscriber ready, listening to /joint_states")

    # -------------------------------------------------------------------

    def _joint_state_callback(self, msg: JointState):
        """Process incoming joint state: compute torques and publish."""
        # ------------------------------------------------------------------
        # Step 1: Extract and reorder joints by name
        # ------------------------------------------------------------------
        name_to_idx = {name: i for i, name in enumerate(msg.name)}

        # Check all required joints are present
        missing = [j for j in JOINT_NAMES if j not in name_to_idx]
        if missing:
            self.get_logger().warn_once(
                f"JointState missing joints: {missing}. Skipping.")
            return

        indices = [name_to_idx[j] for j in JOINT_NAMES]

        q    = np.array([msg.position[i] for i in indices], dtype=float)
        qdot = np.array([msg.velocity[i] for i in indices], dtype=float)

        # ROS time in seconds (float)
        ros_time = (msg.header.stamp.sec
                    + msg.header.stamp.nanosec * 1e-9)

        # ------------------------------------------------------------------
        # Step 2: Estimate qddot via finite difference
        # ------------------------------------------------------------------
        if (self.prev_qdot is not None and self.prev_time is not None):
            dt = ros_time - self.prev_time
            if dt > 1e-9:
                qddot = (qdot - self.prev_qdot) / dt
            else:
                qddot = np.zeros(6)
        else:
            qddot = np.zeros(6)

        self.prev_qdot = qdot.copy()
        self.prev_time = ros_time

        # ------------------------------------------------------------------
        # Step 3: Compute torques and EE acceleration via dynamics model
        # ------------------------------------------------------------------
        try:
            tau  = compute_torques(q, qdot, qddot, self.params)
            a_ee = ee_acceleration(q, qdot, qddot, self.params)
        except Exception as e:
            self.get_logger().error(f"Dynamics computation failed: {e}")
            return

        a_ee_lin = a_ee[:3]

        # ------------------------------------------------------------------
        # Step 4: Update running windows
        # ------------------------------------------------------------------
        self.tau_window.append(np.linalg.norm(tau))
        self.accel_window.append(np.linalg.norm(a_ee_lin))

        # ------------------------------------------------------------------
        # Step 5: Publish
        # ------------------------------------------------------------------
        self.torque_pub.publish(_float64_multiarray(tau))
        self.accel_pub.publish(_float64_multiarray(a_ee_lin))

        if len(self.tau_window) >= 5:
            peak_tau   = float(max(self.tau_window))
            rms_tau    = float(np.sqrt(np.mean(np.array(self.tau_window) ** 2)))
            mean_accel = float(np.mean(list(self.accel_window)))
            self.summary_pub.publish(
                _float64_multiarray([peak_tau, rms_tau, mean_accel]))

        # ------------------------------------------------------------------
        # Step 6: Log to CSV
        # ------------------------------------------------------------------
        row = ([ros_time]
               + q.tolist()
               + qdot.tolist()
               + qddot.tolist()
               + tau.tolist()
               + a_ee_lin.tolist())
        self.csv_writer.writerow(row)

        self.get_logger().debug(
            f"τ_norm={np.linalg.norm(tau):.3f} Nm  "
            f"|a_ee|={np.linalg.norm(a_ee_lin):.3f} m/s²")

    def destroy_node(self):
        """Flush and close CSV on shutdown."""
        self.csv_file.flush()
        self.csv_file.close()
        self.get_logger().info("Metrics CSV closed.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MetricsSubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
