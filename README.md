# ECE383 Final Project — Dynamic Trajectory Optimization for Bartender Robot Arm

Kinova Gen3 Lite 6-DOF arm simulation with mixing-motion trajectory optimization.

**Objective:** Find joint trajectories that maximize end-effector acceleration (mixing effectiveness) while minimizing joint torques (mechanical safety), subject to joint position, velocity, and torque limits.

---

## Repository Structure

```
ECE383-Final-Project/
├── bartender_arm/              # Core Python library (no ROS2 deps — runs on MacBook)
│   ├── kinematics.py           # Forward kinematics + geometric Jacobian
│   ├── dynamics.py             # Recursive Newton-Euler: τ = M(q)q̈ + C(q,q̇)q̇ + G(q)
│   ├── trajectory.py           # Fourier-series trajectory parameterization
│   ├── optimizer.py            # SLSQP constrained optimization
│   └── nodes/                  # ROS2 node scripts (only run inside Docker)
│       ├── trajectory_publisher_node.py
│       └── metrics_subscriber_node.py
├── scripts/
│   └── run_optimization.py     # Standalone offline script — runs on MacBook
├── launch/
│   ├── simulation.launch.py    # Full Gazebo + RViz + kortex + custom nodes
│   └── optimization.launch.py  # Offline optimization via ROS2 launch
├── config/
│   └── robot_params.yaml       # All robot parameters (single source of truth)
└── rviz/
    └── bartender_arm.rviz      # RViz2 config (fixed frame: base_link)
```

---

## Part 1 — Run Offline Optimization on MacBook (No Docker Needed)

The core library and optimization script have no ROS2 dependencies.

### Install Python dependencies

```bash
pip3 install numpy scipy matplotlib pyyaml
```

### Run

```bash
cd ECE383-Final-Project
python3 scripts/run_optimization.py
```

**Output files (written to current directory):**
- `comparison_plots.png` — 4-panel figure (torques, velocities, EE acceleration, EE path)
- `baseline_trajectory.csv` — naive baseline trajectory + torques
- `optimized_trajectory.csv` — optimized trajectory + torques
- `optimized_coeffs.npy` — optimal Fourier coefficients (used by the trajectory publisher node)

### Options

```bash
python3 scripts/run_optimization.py --no-optimize      # evaluate baseline only (fast)
python3 scripts/run_optimization.py --load-coeffs optimized_coeffs.npy  # skip re-optimization
python3 scripts/run_optimization.py --output-dir ./results/
```

---

## Part 2 — Sync to School Machine via Git

My changes are on the Mac. Push them so the Docker container on the school machine gets them.

**On Mac:**
```bash
cd /Users/teresaliang/Desktop/ECE383/ECE383-Final-Project
git add -A
git commit -m "your message"
git push
```

**On the school Ubuntu machine (outside Docker):**
```bash
cd ~/workspaces/ECE383-Final-Project   # or wherever the repo lives on the host
git pull
```

---

## Part 3 — Docker Container Setup

### Start the container

```bash
# On the school Ubuntu machine:
xhost +local:docker

docker run --rm -it \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/workspaces:/root/workspaces \
  --name ros2_lab \
  gitlab-registry.oit.duke.edu/introtorobotics/mems-robotics-toolkit:kinova-jazzy-latest bash
```

Inside the container, open a multi-pane terminal:
```bash
terminator &
```

### Create symlink (first time only)

The project lives at `~/workspaces/ECE383-Final-Project/` on the host (mounted at the same path inside Docker). Colcon needs it under a `src/` directory:

```bash
mkdir -p ~/workspaces/src
ln -sfn ~/workspaces/ECE383-Final-Project ~/workspaces/src/bartender_arm
```

Verify:
```bash
ls -la ~/workspaces/src/bartender_arm   # should show the symlink target
```

### Install Python dependencies (first time only)

```bash
pip3 install scipy numpy matplotlib pyyaml
```

### Build

The Docker image already has `kortex_description`, `kortex_bringup`, and other Kinova packages pre-installed. Only build the custom package:

```bash
cd ~/workspaces
source /opt/ros/jazzy/setup.bash
colcon build --packages-select bartender_arm
source install/setup.bash
```

Add `source install/setup.bash` to `~/.bashrc` so you don't have to repeat it in every pane:

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo "source /root/workspaces/install/setup.bash" >> ~/.bashrc
```

### Copy optimized coefficients into the container (if you ran optimization on Mac)

Transfer `optimized_coeffs.npy` from Mac to the school machine, then place it where the node will find it:

```bash
cp /path/to/optimized_coeffs.npy \
   ~/workspaces/install/bartender_arm/share/bartender_arm/optimized_coeffs.npy
```

---

## Part 4 — Running the Simulation

Source in every new terminal pane before running anything:
```bash
source /opt/ros/jazzy/setup.bash
source ~/workspaces/install/setup.bash
```

### Option A: One-command launch

```bash
ros2 launch bartender_arm simulation.launch.py                      # optimized (default)
ros2 launch bartender_arm simulation.launch.py trajectory_mode:=baseline
```

This starts Gazebo, RViz (5s delay), metrics subscriber, and trajectory publisher (8s delay) automatically.

### Option B: Manual launch (for debugging — use separate Terminator panes)

**Pane 1 — Gazebo + controller:**
```bash
ros2 launch kortex_bringup kortex_sim_control.launch.py \
  robot_type:=gen3_lite dof:=6 sim_gazebo:=true \
  gripper:=gen3_lite_2f launch_rviz:=false
```

**Pane 2 — RViz (wait ~5s for Gazebo):**
```bash
ros2 run rviz2 rviz2 \
  -d ~/workspaces/install/bartender_arm/share/bartender_arm/rviz/bartender_arm.rviz
```
> If RViz shows **"Frame [map] does not exist"**, change Fixed Frame to `base_link` in the left panel under Global Options. Our `.rviz` config sets this automatically.

**Pane 3 — Metrics subscriber:**
```bash
ros2 run bartender_arm metrics_subscriber_node
```
> No need to pass `--ros-args -p config_path:=...` — the node auto-detects the installed config via `get_package_share_directory`.

**Pane 4 — Trajectory publisher (wait ~8s for controller):**
```bash
ros2 run bartender_arm trajectory_publisher_node \
  --ros-args -p trajectory_mode:=optimized
```

### Monitor metrics in real-time

```bash
ros2 topic echo /bartender/joint_torques
ros2 topic echo /bartender/ee_acceleration
ros2 topic echo /bartender/metrics_summary    # [peak_torque, rms_torque, mean_ee_accel]
cat /tmp/bartender_metrics.csv                # full log
```

---

## Engineering Design

### Trajectory Parameterization

Fourier series over 6 joints with K=3 harmonics:

```
q_i(t) = q0_i + Σ_{k=1}^{3} [ a_{i,k}·sin(k·2πf·t) + b_{i,k}·cos(k·2πf·t) ]
```

36 optimization variables. Analytic derivatives — no finite differences during optimization.

### Dynamics Model

Full Recursive Newton-Euler using inertial parameters from `kortex_description` URDF:

```
τ = M(q)q̈ + C(q,q̇)q̇ + G(q)
```

Verified: `M(q)@qddot + C + G == rne(q,qdot,qddot)` to machine precision (< 1e-15 error).

### Optimization

```
Minimize:  J(x) = (1/N) Σ_t [ 1.0·||τ(t)||² − 0.5·||a_ee(t)||² ]

Subject to:
    q_min[i]  ≤ q_i(t)   ≤ q_max[i]    (joint limits)
    |q̇_i(t)|             ≤ v_max[i]    (velocity limits)
    |τ_i(t)|              ≤ τ_max[i]    (torque limits)
    Σ x²                  ≥ 0.01        (non-trivial motion)
```

Solver: `scipy.optimize.minimize` with `method='SLSQP'`, ~900 inequality constraints.

---

## Grading Checklist (A-Level)

- [x] Full dynamic torque computation — Recursive Newton-Euler
- [x] Defined optimization objective — mixed torque + acceleration
- [x] Constrained SLSQP optimization
- [x] Comparative plots — baseline vs optimized (torques, velocities, EE acceleration, EE path)
- [x] Constraint satisfaction verified — position, velocity, torque limits
- [x] ROS2 Gazebo + RViz simulation with trajectory execution
- [x] Real-time metrics monitoring and CSV logging
