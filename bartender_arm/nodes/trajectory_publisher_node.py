#!/usr/bin/env python3
"""
trajectory_publisher_node.py — ROS2 node that publishes a pre-computed
mixing trajectory to the Kinova Gen3 Lite joint trajectory controller.

Publishes to:
    /joint_trajectory_controller/joint_trajectory
    (trajectory_msgs/msg/JointTrajectory)

Also publishes an EE path marker for RViz:
    /bartender/ee_path  (visualization_msgs/msg/Marker)

Parameters:
    trajectory_mode  (str)  : 'optimized' or 'baseline'  [default: 'optimized']
    loop_trajectory  (bool) : re-publish every cycle      [default: true]
    config_path      (str)  : path to robot_params.yaml   [default: package share]
    coeffs_path      (str)  : path to optimized_coeffs.npy
"""

import os
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

from bartender_arm.kinematics import forward_kinematics
from bartender_arm.trajectory import make_trajectory, baseline_params, sample_trajectory


JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3',
               'joint_4', 'joint_5', 'joint_6']


class TrajectoryPublisherNode(Node):
    def __init__(self):
        super().__init__('trajectory_publisher')

        # ---------------------------------------------------------------
        # Declare parameters
        # ---------------------------------------------------------------
        self.declare_parameter('trajectory_mode', 'optimized')
        self.declare_parameter('loop_trajectory', True)
        self.declare_parameter('config_path', '')
        self.declare_parameter('coeffs_path', '')

        mode       = self.get_parameter('trajectory_mode').value
        loop       = self.get_parameter('loop_trajectory').value
        cfg_path   = self.get_parameter('config_path').value
        coeff_path = self.get_parameter('coeffs_path').value

        # ---------------------------------------------------------------
        # Resolve config path via ament index (works installed or symlinked)
        # ---------------------------------------------------------------
        if not cfg_path:
            share_dir = get_package_share_directory('bartender_arm')
            cfg_path  = os.path.join(share_dir, 'config', 'robot_params.yaml')

        if not os.path.isfile(cfg_path):
            self.get_logger().error(f'Config not found: {cfg_path}')
            raise FileNotFoundError(cfg_path)

        self.get_logger().info(f'Loading config from: {cfg_path}')
        with open(cfg_path) as f:
            self.params = yaml.safe_load(f)

        # ---------------------------------------------------------------
        # Build trajectory
        # ---------------------------------------------------------------
        traj_cfg   = self.params['trajectory']
        self.f     = float(traj_cfg['frequency'])
        duration   = float(traj_cfg['duration'])
        n_wp       = int(traj_cfg['n_waypoints'])
        n_harm     = int(traj_cfg['n_harmonics'])
        self.q0    = np.array(traj_cfg['mixing_center_joints'], dtype=float)

        t_dense = np.linspace(0.0, duration, n_wp * 4, endpoint=False)

        if mode == 'optimized':
            if not coeff_path:
                # Look next to the package share dir (user copied it there)
                share_dir  = get_package_share_directory('bartender_arm')
                coeff_path = os.path.join(share_dir, 'optimized_coeffs.npy')
                # Also check workspace root
                if not os.path.isfile(coeff_path):
                    coeff_path = os.path.join(
                        os.path.dirname(share_dir), '..', '..', '..', 'optimized_coeffs.npy')
                    coeff_path = os.path.normpath(coeff_path)

            if os.path.isfile(coeff_path):
                x = np.load(coeff_path)
                self.get_logger().info(f'Loaded optimized coefficients from {coeff_path}')
            else:
                self.get_logger().warn(
                    'optimized_coeffs.npy not found — using baseline. '
                    'Run scripts/run_optimization.py first and copy the .npy file here.')
                x = baseline_params(n_joints=6, n_harmonics=n_harm)
        else:
            x = baseline_params(n_joints=6, n_harmonics=n_harm)
            self.get_logger().info('Using baseline trajectory')

        traj_dense  = make_trajectory(x, self.f, t_dense, self.q0, n_harmonics=n_harm)
        self.traj   = sample_trajectory(traj_dense, n_wp)
        self.duration = duration

        # ---------------------------------------------------------------
        # Publishers
        # ---------------------------------------------------------------
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            qos_profile=10,
        )
        self.marker_pub = self.create_publisher(
            Marker, '/bartender/ee_path', qos_profile=10)

        self.get_logger().info(
            f'TrajectoryPublisher ready — mode={mode}, '
            f'waypoints={len(self.traj["t"])}, duration={duration}s, loop={loop}')

        self._publish_trajectory()
        self._publish_ee_path_marker()

        if loop:
            self.timer = self.create_timer(duration, self._on_timer)

    def _on_timer(self):
        self._publish_trajectory()

    def _publish_trajectory(self):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names  = JOINT_NAMES

        q     = self.traj['q']
        qdot  = self.traj['qdot']
        t_arr = self.traj['t']
        t_offset = 0.5  # give controller time to receive before first waypoint

        last_idx = len(t_arr) - 1
        for idx in range(len(t_arr)):
            pt = JointTrajectoryPoint()
            pt.positions  = q[idx].tolist()
            pt.velocities = qdot[idx].tolist() if idx < last_idx else [0.0] * 6
            t_sec   = float(t_arr[idx]) + t_offset
            sec_int = int(t_sec)
            nanosec = int(round((t_sec - sec_int) * 1e9))
            pt.time_from_start = Duration(sec=sec_int, nanosec=nanosec)
            msg.points.append(pt)

        self.traj_pub.publish(msg)
        self.get_logger().debug(f'Published {len(msg.points)} waypoints')

    def _publish_ee_path_marker(self):
        marker = Marker()
        marker.header.frame_id    = 'base_link'
        marker.header.stamp       = self.get_clock().now().to_msg()
        marker.ns                 = 'ee_path'
        marker.id                 = 0
        marker.type               = Marker.LINE_STRIP
        marker.action             = Marker.ADD
        marker.scale.x            = 0.005
        marker.color              = ColorRGBA(r=0.0, g=0.8, b=0.2, a=0.9)
        marker.pose.orientation.w = 1.0

        for idx in range(len(self.traj['q'])):
            _, T_ee = forward_kinematics(self.traj['q'][idx], self.params)
            p = T_ee[:3, 3]
            marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))

        if marker.points:
            marker.points.append(marker.points[0])  # close the loop

        self.marker_pub.publish(marker)
        self.get_logger().info('Published EE path marker')


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
