"""
simulation.launch.py — Full simulation stack for the Kinova Gen3 Lite
bartender arm trajectory optimization project.

Launches (in order):
    1. kortex_sim_control.launch.py  — Gazebo + ros2_control + kortex URDF
    2. RViz2 with bartender_arm.rviz config  (5s delay)
    3. metrics_subscriber_node               (immediate)
    4. trajectory_publisher_node             (8s delay, after controller init)

Prerequisites (inside the Docker container):
    source /opt/ros/jazzy/setup.bash
    cd ~/ros2_ws && colcon build --packages-select kortex_description \
                                                   kortex_bringup \
                                                   bartender_arm
    source install/setup.bash

Usage:
    # Default: publish optimized trajectory
    ros2 launch bartender_arm simulation.launch.py

    # Use baseline instead
    ros2 launch bartender_arm simulation.launch.py trajectory_mode:=baseline

    # Both modes for comparison (two separate terminals)
    ros2 launch bartender_arm simulation.launch.py trajectory_mode:=baseline
    ros2 launch bartender_arm simulation.launch.py trajectory_mode:=optimized
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    TimerAction,
    DeclareLaunchArgument,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # -----------------------------------------------------------------------
    # Declare launch arguments
    # -----------------------------------------------------------------------
    trajectory_mode_arg = DeclareLaunchArgument(
        'trajectory_mode',
        default_value='optimized',
        description='Trajectory to publish: "optimized" or "baseline"',
        choices=['optimized', 'baseline'],
    )

    # -----------------------------------------------------------------------
    # Resolve package share directories
    # -----------------------------------------------------------------------
    bartender_share  = FindPackageShare('bartender_arm')
    kortex_bringup   = get_package_share_directory('kortex_bringup')

    config_path = PathJoinSubstitution([bartender_share, 'config', 'robot_params.yaml'])
    rviz_config = PathJoinSubstitution([bartender_share, 'rviz', 'bartender_arm.rviz'])

    # -----------------------------------------------------------------------
    # 1. Kortex simulation (Gazebo + ros2_control + robot_description)
    # -----------------------------------------------------------------------
    kortex_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(kortex_bringup, 'launch', 'kortex_sim_control.launch.py')
        ),
        launch_arguments={
            'robot_type':  'gen3_lite',
            'dof':         '6',
            'sim_gazebo':  'true',
            'gripper':     'gen3_lite_2f',
            'launch_rviz': 'false',   # we launch our own RViz
        }.items(),
    )

    log_kortex = LogInfo(msg="[1/4] Launched kortex_sim_control (Gazebo + controller)")

    # -----------------------------------------------------------------------
    # 2. Metrics subscriber node (starts immediately)
    # -----------------------------------------------------------------------
    metrics_node = Node(
        package='bartender_arm',
        executable='metrics_subscriber_node',
        name='metrics_subscriber',
        parameters=[
            config_path,
            {'log_path': '/tmp/bartender_metrics.csv'},
        ],
        output='screen',
        emulate_tty=True,
    )

    log_metrics = LogInfo(msg="[2/4] Started metrics_subscriber_node")

    # -----------------------------------------------------------------------
    # 3. RViz2 with custom config (5s delay — wait for robot_description)
    # -----------------------------------------------------------------------
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
    )

    rviz_delayed = TimerAction(
        period=5.0,
        actions=[
            LogInfo(msg="[3/4] Launching RViz2"),
            rviz_node,
        ],
    )

    # -----------------------------------------------------------------------
    # 4. Trajectory publisher node (8s delay — wait for controller ready)
    # -----------------------------------------------------------------------
    traj_node = Node(
        package='bartender_arm',
        executable='trajectory_publisher_node',
        name='trajectory_publisher',
        parameters=[
            config_path,
            {
                'trajectory_mode': LaunchConfiguration('trajectory_mode'),
                'loop_trajectory': True,
                # coeffs_path defaults to repo root / optimized_coeffs.npy
            },
        ],
        output='screen',
        emulate_tty=True,
    )

    traj_delayed = TimerAction(
        period=8.0,
        actions=[
            LogInfo(msg="[4/4] Launching trajectory_publisher_node"),
            traj_node,
        ],
    )

    # -----------------------------------------------------------------------
    # Assemble
    # -----------------------------------------------------------------------
    return LaunchDescription([
        trajectory_mode_arg,
        kortex_sim,
        log_kortex,
        metrics_node,
        log_metrics,
        rviz_delayed,
        traj_delayed,
    ])
