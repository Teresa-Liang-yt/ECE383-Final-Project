from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'bartender_arm'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ROS2 package resource index
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        # package.xml
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        # Config files
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
        # RViz config
        (os.path.join('share', package_name, 'rviz'),
         glob('rviz/*.rviz')),
        # Scripts (also installed as executables via console_scripts)
        (os.path.join('share', package_name, 'scripts'),
         glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ECE383 Student',
    maintainer_email='student@example.com',
    description='Mixing trajectory optimization for Kinova Gen3 Lite',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_publisher_node = scripts.trajectory_publisher_node:main',
            'metrics_subscriber_node   = scripts.metrics_subscriber_node:main',
        ],
    },
)
