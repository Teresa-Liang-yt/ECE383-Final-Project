# ECE383 Final Project — Dynamic Trajectory Optimization for Bartender Robot Arm

Kinova Gen3 Lite 6-DOF arm simulation with mixing-motion trajectory optimization.

**Objective:** Find joint trajectories that maximize end-effector acceleration (mixing effectiveness) while minimizing joint torques (mechanical safety), subject to joint position, velocity, and torque constraints.

## Repository Structure

```
bartender_arm/
├── bartender_arm/          # Core Python library (no ROS2 deps — runs on MacBook)
│   ├── kinematics.py       # Forward kinematics + geometric Jacobian (URDF transform chain)
│   ├── dynamics.py         # Recursive Newton-Euler: τ = M(q)q̈ + C(q,q̇)q̇ + G(q)
│   ├── trajectory.py       # Fourier-series trajectory parameterization
│   └── optimizer.py        # SLSQP constrained optimization
├── scripts/
│   ├── run_optimization.py         # Standalone offline script — runs on MacBook
│   ├── trajectory_publisher_node.py # ROS2 node: publishes trajectory to controller
│   └── metrics_subscriber_node.py  # ROS2 node: computes real-time torques/metrics
├── launch/
│   ├── simulation.launch.py        # Full Gazebo + RViz + kortex + custom nodes
│   └── optimization.launch.py      # Offline optimization via ROS2 launch
├── config/
│   └── robot_params.yaml           # All robot parameters (URDF-derived, single source of truth)
└── rviz/
    └── bartender_arm.rviz          # RViz2 configuration
```

---

## Part 1 — Run Offline Optimization on MacBook (No Docker Needed)

The core library and optimization script have no ROS2 dependencies. You can run everything locally.

### Install dependencies

```bash
pip3 install numpy scipy matplotlib pyyaml
```

### Run optimization

```bash
cd ECE383-Final-Project
python3 scripts/run_optimization.py
```

**Output:**
- `comparison_plots.png` — 4-panel figure (torques, velocities, EE acceleration, EE path)
- `baseline_trajectory.csv` — naive baseline trajectory + torques
- `optimized_trajectory.csv` — optimized trajectory + torques
- `optimized_coeffs.npy` — optimal Fourier coefficients (used by ROS2 nodes)

### Run with options

```bash
# Skip optimization, just evaluate baseline:
python3 scripts/run_optimization.py --no-optimize

# Load previously saved coefficients instead of re-running optimization:
python3 scripts/run_optimization.py --load-coeffs optimized_coeffs.npy

# Save outputs to a specific directory:
python3 scripts/run_optimization.py --output-dir ./results/
```

---

## Part 2 — Simulation in Docker Container

### Course Docker image

```bash
# On your school Ubuntu machine:
xhost +local:docker

docker run --rm -it \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/workspaces:/root/workspaces \
  --name ros2_lab \
  gitlab-registry.oit.duke.edu/introtorobotics/mems-robotics-toolkit:kinova-jazzy-latest bash
```

Once inside the container, launch a multi-pane terminal:

```bash
terminator &
```

### Workspace setup (first time only)

The project must live at `/root/workspaces/src/bartender_arm/` inside the container.  
Transfer from your Mac or clone from your Git repo:

```bash
# Inside container — create workspace if it doesn't exist:
mkdir -p /root/workspaces/src
cd /root/workspaces

# Option A: if you have the project on the school machine mounted at ~/workspaces:
# Just copy it there before starting the container.

# Option B: clone from Git:
# git clone <your-repo-url> src/bartender_arm
```

### Check if Kinova packages are pre-installed

The course image is named `kinova-jazzy` and likely has `kortex_description` and `kortex_bringup` pre-installed. Check:

```bash
ros2 pkg list | grep kortex
```

**If packages are found** (pre-installed): skip the clone step below.

**If packages are NOT found**: clone ros2_kortex alongside your package:

```bash
cd /root/workspaces
git clone https://github.com/Kinovarobotics/ros2_kortex.git src/ros2_kortex -b main
rosdep install --from-paths src --ignore-src -r -y
```

### Install Python dependencies

```bash
pip3 install scipy numpy matplotlib pyyaml
```

### Build

```bash
cd /root/workspaces
source /opt/ros/jazzy/setup.bash

# If kortex is pre-installed, only build our package:
colcon build --packages-select bartender_arm

# If you cloned ros2_kortex:
colcon build --packages-select kortex_description kortex_bringup bartender_arm

source install/setup.bash
```

### Copy optimized coefficients (if you ran optimization on MacBook)

```bash
# Transfer optimized_coeffs.npy from MacBook to school machine, then:
cp optimized_coeffs.npy /root/workspaces/
# The trajectory_publisher_node will find it there by default.
```

---

## Part 3 — Running the Simulation

Open 4 panes in Terminator. In each:

```bash
source /opt/ros/jazzy/setup.bash
source /root/workspaces/install/setup.bash
```

### Option A: One-command launch (recommended)

```bash
# Pane 1 — full simulation stack:
ros2 launch bartender_arm simulation.launch.py

# This launches Gazebo, RViz, metrics_subscriber, and trajectory_publisher
# automatically with appropriate delays (5s for RViz, 8s for trajectory).
```

### Option B: Manual launch (for debugging)

```bash
# Pane 1 — Gazebo + ros2_control:
ros2 launch kortex_bringup kortex_sim_control.launch.py \
  robot_type:=gen3_lite dof:=6 sim_gazebo:=true \
  gripper:=gen3_lite_2f launch_rviz:=false

# Pane 2 — RViz (wait ~5s for Gazebo to start):
ros2 run rviz2 rviz2 -d /root/workspaces/install/bartender_arm/share/bartender_arm/rviz/bartender_arm.rviz

# Pane 3 — Metrics subscriber:
ros2 run bartender_arm metrics_subscriber_node \
  --ros-args -p config_path:=/root/workspaces/src/bartender_arm/config/robot_params.yaml

# Pane 4 — Trajectory publisher (wait ~8s for controller to initialize):
ros2 run bartender_arm trajectory_publisher_node \
  --ros-args -p trajectory_mode:=optimized
```

### Switch between trajectories

```bash
# Run with baseline:
ros2 launch bartender_arm simulation.launch.py trajectory_mode:=baseline

# Run with optimized:
ros2 launch bartender_arm simulation.launch.py trajectory_mode:=optimized
```

### Monitor metrics

```bash
# Watch torques in real-time:
ros2 topic echo /bartender/joint_torques

# Watch EE acceleration:
ros2 topic echo /bartender/ee_acceleration

# Summary stats [peak_torque, rms_torque, mean_ee_accel]:
ros2 topic echo /bartender/metrics_summary

# Metrics are also logged to:
cat /tmp/bartender_metrics.csv
```

---

## Engineering Design

### Trajectory Parameterization

Fourier series over 6 joints with K=3 harmonics:

```
q_i(t) = q0_i + Σ_{k=1}^{3} [ a_{i,k}·sin(k·2πf·t) + b_{i,k}·cos(k·2πf·t) ]
```

This gives **36 optimization variables** with analytic derivatives (no finite differences needed during optimization).

### Dynamics Model

Full Recursive Newton-Euler algorithm using inertial parameters extracted from the `kortex_description` URDF:

```
τ = M(q)q̈ + C(q,q̇)q̇ + G(q)
```

Verified: `M(q)@qddot + C(q,qdot) + G(q) == rne(q,qdot,qddot)` to machine precision (error < 1e-15).

### Optimization Problem

```
Minimize:  J(x) = (1/N) Σ_t [ 1.0 · ||τ(t)||² - 0.5 · ||a_ee(t)||² ]

Subject to:
    q_min[i]  ≤ q_i(t)    ≤ q_max[i]    ∀ i, t
    |q̇_i(t)|             ≤ v_max[i]    ∀ i, t
    |τ_i(t)|              ≤ τ_max[i]    ∀ i, t
    Σ x²                  ≥ 0.01        (non-trivial motion)
```

Solver: `scipy.optimize.minimize` with `method='SLSQP'`.

### Baseline Trajectory

Simple sinusoidal oscillation of joint_2 at 1 Hz (amplitude 0.18 rad). Represents naive "just move the arm" approach.

---

## Verification

```bash
# Run just the verification checks (no optimization):
python3 scripts/run_optimization.py --no-optimize --skip-verification
python3 -c "
import sys; sys.path.insert(0, '.')
from bartender_arm.kinematics import verify_jacobian_numerically
from bartender_arm.dynamics import verify_dynamics_consistency
import numpy as np, yaml
params = yaml.safe_load(open('config/robot_params.yaml'))
print('Jacobian error:', verify_jacobian_numerically(np.zeros(6), params))
print('Dynamics error:', verify_dynamics_consistency(np.zeros(6), np.zeros(6), np.zeros(6), params))
"
```

Expected output:
- Jacobian error: < 3e-7
- Dynamics error: < 1e-14

---

## Grading Checklist (A-Level)

- [x] Full dynamic torque computation via Recursive Newton-Euler
- [x] Defined optimization objective function (mixed torque + acceleration)
- [x] Constrained optimization implemented (SLSQP, ~900 inequality constraints)
- [x] Comparative plots: baseline vs optimized (torque, velocity, EE acceleration, EE path)
- [x] Constraint satisfaction verified (joint limits, velocity limits, torque limits)
- [x] ROS2 Gazebo simulation with trajectory execution
- [x] Real-time metrics monitoring
