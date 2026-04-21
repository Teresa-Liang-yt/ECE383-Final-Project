#!/usr/bin/env python3
"""
metrics_subscriber_node.py — ROS2 node that listens to /joint_states,
computes real-time joint torques and EE acceleration via the dynamics model,
and publishes the results for monitoring and logging.

Subscribes to:
    /joint_states  (sensor_msgs/msg/JointState)

Publishes:
    /bartender/joint_torques    (std_msgs/msg/Float64MultiArray)
    /bartender/ee_acceleration  (std_msgs/msg/Float64MultiArray)
    /bartender/metrics_summary  (std_msgs/msg/Float64MultiArray) — [peak_tau, rms_tau, mean_accel]

Logs all data to /tmp/bartender_metrics.csv.
"""

import os
import csv
from collections import deque

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

from bartender_arm.dynamics import compute_torques
from bartender_arm.kinematics import ee_acceleration


JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3',
               'joint_4', 'joint_5', 'joint_6']


def _float64_multiarray(values):
    msg = Float64MultiArray()
    msg.layout.dim.append(
        MultiArrayDimension(label='data', size=len(values), stride=len(values)))
    msg.data = [float(v) for v in values]
    return msg


class MetricsSubscriberNode(Node):
    def __init__(self):
        super().__init__('metrics_subscriber')

        self.declare_parameter('config_path', '')
        self.declare_parameter('log_path', '/tmp/bartender_metrics.csv')
        self.declare_parameter('window_size', 100)

        cfg_path    = self.get_parameter('config_path').value
        log_path    = self.get_parameter('log_path').value
        window_size = int(self.get_parameter('window_size').value)

        if not cfg_path:
            share_dir = get_package_share_directory('bartender_arm')
            cfg_path  = os.path.join(share_dir, 'config', 'robot_params.yaml')

        self.get_logger().info(f'Loading config from: {cfg_path}')
        with open(cfg_path) as f:
            self.params = yaml.safe_load(f)

        # Sliding window of (time, qdot) pairs for smoothed FD acceleration estimate.
        # Using the oldest and newest points spans 2 steps, halving derivative noise.
        self.qdot_history = deque(maxlen=3)
        self.tau_window   = deque(maxlen=window_size)
        self.accel_window = deque(maxlen=window_size)

        # CSV logger
        self.csv_file   = open(log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'ros_time',
            'q1','q2','q3','q4','q5','q6',
            'qdot1','qdot2','qdot3','qdot4','qdot5','qdot6',
            'qddot1','qddot2','qddot3','qddot4','qddot5','qddot6',
            'tau1','tau2','tau3','tau4','tau5','tau6',
            'ee_ax','ee_ay','ee_az',
        ])
        self.get_logger().info(f'Logging metrics to {log_path}')

        self.sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, 10)
        self.torque_pub  = self.create_publisher(
            Float64MultiArray, '/bartender/joint_torques',   10)
        self.accel_pub   = self.create_publisher(
            Float64MultiArray, '/bartender/ee_acceleration', 10)
        self.summary_pub = self.create_publisher(
            Float64MultiArray, '/bartender/metrics_summary', 10)

        self.get_logger().info('MetricsSubscriber ready, listening to /joint_states')

    def _joint_state_callback(self, msg: JointState):
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        missing = [j for j in JOINT_NAMES if j not in name_to_idx]
        if missing:
            self.get_logger().warn_once(f'JointState missing joints: {missing}')
            return

        indices = [name_to_idx[j] for j in JOINT_NAMES]
        q    = np.array([msg.position[i] for i in indices], dtype=float)
        qdot = np.array([msg.velocity[i] for i in indices], dtype=float)
        ros_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Estimate qddot via FD over a 3-sample window (endpoints span 2 steps,
        # which smooths single-sample noise without adding latency).
        self.qdot_history.append((ros_time, qdot.copy()))
        if len(self.qdot_history) >= 2:
            t0, qdot0 = self.qdot_history[0]
            t1, qdot1 = self.qdot_history[-1]
            dt = t1 - t0
            qddot = (qdot1 - qdot0) / dt if dt > 1e-9 else np.zeros(6)
        else:
            qddot = np.zeros(6)

        try:
            tau  = compute_torques(q, qdot, qddot, self.params)
            a_ee = ee_acceleration(q, qdot, qddot, self.params)
        except Exception as e:
            self.get_logger().error(f'Dynamics failed: {e}')
            return

        a_ee_lin = a_ee[:3]
        self.tau_window.append(np.linalg.norm(tau))
        self.accel_window.append(np.linalg.norm(a_ee_lin))

        self.torque_pub.publish(_float64_multiarray(tau))
        self.accel_pub.publish(_float64_multiarray(a_ee_lin))

        if len(self.tau_window) >= 5:
            peak_tau   = float(max(self.tau_window))
            rms_tau    = float(np.sqrt(np.mean(np.array(self.tau_window) ** 2)))
            mean_accel = float(np.mean(list(self.accel_window)))
            self.summary_pub.publish(_float64_multiarray([peak_tau, rms_tau, mean_accel]))

        row = ([ros_time] + q.tolist() + qdot.tolist()
               + qddot.tolist() + tau.tolist() + a_ee_lin.tolist())
        self.csv_writer.writerow(row)

        self.get_logger().debug(
            f'τ_norm={np.linalg.norm(tau):.3f} Nm  '
            f'|a_ee|={np.linalg.norm(a_ee_lin):.3f} m/s²')

    def destroy_node(self):
        self.csv_file.flush()
        self.csv_file.close()
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
