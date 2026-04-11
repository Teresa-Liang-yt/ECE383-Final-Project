"""
optimization.launch.py — Lightweight launch for offline optimization only.

No Gazebo or RViz. Runs run_optimization.py as a subprocess and streams
its output to the terminal.

Usage:
    ros2 launch bartender_arm optimization.launch.py
    ros2 launch bartender_arm optimization.launch.py output_dir:=/tmp/results
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='.',
        description='Directory to write output plots and CSVs',
    )

    bartender_share = FindPackageShare('bartender_arm')
    script_path     = PathJoinSubstitution([bartender_share, 'scripts', 'run_optimization.py'])
    config_path     = PathJoinSubstitution([bartender_share, 'config', 'robot_params.yaml'])

    opt_process = ExecuteProcess(
        cmd=[
            'python3', script_path,
            '--config',     config_path,
            '--output-dir', LaunchConfiguration('output_dir'),
        ],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        output_dir_arg,
        opt_process,
    ])
