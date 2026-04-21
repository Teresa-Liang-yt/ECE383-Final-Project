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

**Expected results (verified):**
- RMS torque: −34.5% (from 5.70 → 3.73 Nm)
- Mean EE acceleration: +60.3% (from 2.76 → 4.43 m/s²)
- Peak EE acceleration: +121.7% (from 4.18 → 9.26 m/s²)
- All constraints satisfied: position, velocity (margin 0.05 rad/s), torque (margin 5.4 Nm)

### Options

```bash
python3 scripts/run_optimization.py --no-optimize      # evaluate baseline only (fast)
python3 scripts/run_optimization.py --load-coeffs optimized_coeffs.npy  # skip re-optimization
python3 scripts/run_optimization.py --output-dir ./results/
```

**Runtime:** ~100 seconds on a MacBook M3.

---

## Part 2 — Sync Code and Coefficients to School Machine

### Push code from Mac

```bash
cd /Users/teresaliang/Desktop/ECE383/ECE383-Final-Project
git add -A
git commit -m "your message"
git push
```

### Pull on the school Ubuntu machine (outside Docker)

```bash
cd ~/workspaces/ECE383-Final-Project
git pull
```

### Transfer optimized coefficients separately

`optimized_coeffs.npy` is listed in `.gitignore` (binary files aren't tracked), so it will **not** sync via git. After running the optimization on Mac, copy it to the school machine manually:

```bash
# Run this on Mac — replace <netid> and <school-machine-hostname>
scp /Users/teresaliang/Desktop/ECE383/ECE383-Final-Project/optimized_coeffs.npy \
    <netid>@<school-machine-hostname>:~/workspaces/ECE383-Final-Project/optimized_coeffs.npy
```

Alternatively, run the optimization directly inside Docker (scipy is available there too) — see the `--no-optimize` / `--load-coeffs` flags in Part 1.

---

## Part 3 — Docker Container Setup

> All steps below run on the **school Ubuntu machine** unless noted.

### Start the container

```bash
# Allow Docker to open GUI windows (do this once per host login session)
xhost +local:docker

docker run --rm -it \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/workspaces:/root/workspaces \
  --name ros2_lab \
  gitlab-registry.oit.duke.edu/introtorobotics/mems-robotics-toolkit:kinova-jazzy-latest bash
```

> `--rm` means the container is deleted when you exit. The `~/workspaces` volume persists on the host, so project files survive, but **anything installed inside the container (e.g. pip packages) is lost on restart** — you'll need to reinstall them each session.

To open a multi-pane terminal inside the container:
```bash
terminator &
```

### One-time setup (inside the container)

#### 1. Create the colcon symlink

Colcon requires source packages to live under a `src/` directory. Create a symlink so it can find the project:

```bash
mkdir -p ~/workspaces/src
ln -sfn ~/workspaces/ECE383-Final-Project ~/workspaces/src/bartender_arm
ls -la ~/workspaces/src/bartender_arm   # verify: should point to ECE383-Final-Project
```

#### 2. Install Python dependencies

```bash
pip3 install scipy numpy matplotlib pyyaml
```

#### 3. Build the package

The Docker image already has `kortex_description`, `kortex_bringup`, and all Kinova packages pre-installed. Only build the custom package:

```bash
cd ~/workspaces
source /opt/ros/jazzy/setup.bash
colcon build --packages-select bartender_arm
source install/setup.bash
```

Add both source lines to `~/.bashrc` so every new pane picks them up automatically:

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo "source /root/workspaces/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

#### 4. Copy optimized coefficients into the install directory

This step must happen **after** `colcon build` (the build creates the install directory). The trajectory publisher looks for the file at this path:

```bash
cp ~/workspaces/ECE383-Final-Project/optimized_coeffs.npy \
   ~/workspaces/install/bartender_arm/share/bartender_arm/optimized_coeffs.npy
```

> If you rebuild with `colcon build`, the install directory is refreshed and this file may be overwritten — re-run the `cp` command after each rebuild.

---

## Part 4 — Running the Simulation

> All commands below run **inside the Docker container**.

Source ROS2 and the workspace overlay in **every new terminal pane** before running anything (skip if you added them to `~/.bashrc` already):

```bash
source /opt/ros/jazzy/setup.bash
source ~/workspaces/install/setup.bash
```

### Option A: One-command launch (recommended)

```bash
ros2 launch bartender_arm simulation.launch.py                       # optimized trajectory (default)
ros2 launch bartender_arm simulation.launch.py trajectory_mode:=baseline
```

This automatically starts Gazebo, then RViz after a 5-second delay, then the trajectory publisher after an 8-second delay. The metrics subscriber starts immediately.

### Option B: Manual launch (for debugging — use separate Terminator panes)

**Pane 1 — Gazebo + ros2_control:**
```bash
ros2 launch kortex_bringup kortex_sim_control.launch.py \
  robot_type:=gen3_lite dof:=6 sim_gazebo:=true launch_rviz:=false
```

**Pane 2 — RViz (wait ~5s for Gazebo to finish loading):**
```bash
ros2 run rviz2 rviz2 \
  -d ~/workspaces/install/bartender_arm/share/bartender_arm/rviz/bartender_arm.rviz
```

**Pane 3 — Metrics subscriber:**
```bash
ros2 run bartender_arm metrics_subscriber_node
```

**Pane 4 — Trajectory publisher (wait ~8s for the controller to be ready):**
```bash
ros2 run bartender_arm trajectory_publisher_node \
  --ros-args -p trajectory_mode:=optimized
```

> If `optimized_coeffs.npy` is missing, the publisher falls back to the baseline trajectory and logs a warning — check the pane output if the arm isn't moving as expected.

### Monitor metrics in real-time

```bash
ros2 topic echo /bartender/joint_torques
ros2 topic echo /bartender/ee_acceleration
ros2 topic echo /bartender/metrics_summary    # [peak_torque, rms_torque, mean_ee_accel]
cat /tmp/bartender_metrics.csv                # full log (inside container)
```

> Real-time `qddot` is estimated via finite difference over a 3-sample sliding window, spanning two time steps to reduce single-sample derivative noise. The offline optimization uses analytic derivatives and is unaffected.

---

## Engineering Design

### Trajectory Parameterization

Fourier series over 6 joints with K=3 harmonics:

```
q_i(t) = q0_i + Σ_{k=1}^{3} [ a_{i,k}·sin(k·2πf·t) + b_{i,k}·cos(k·2πf·t) ]
```

36 optimization variables. Analytic derivatives — no finite differences during optimization.

**Velocity bound by construction:** With coefficient bound `C = 0.026 rad`, the maximum possible joint velocity is `(1+2+3)·2π·√2·C = 53.3·C = 1.385 rad/s < 1.396 rad/s` for all time. No separate velocity constraint is needed in the optimizer.

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
    |τ_i(t)|              ≤ τ_max[i]    (torque limits)
    Σ x²                  ≥ 0.001       (non-trivial motion)
    (velocity limits guaranteed by coefficient bounds — see above)
```

Solver: `scipy.optimize.minimize` with `method='SLSQP'`, 3 batched vector-valued constraint functions (~541 scalar constraints total). Converges in ~20 iterations, ~100 seconds.

---

## Grading Checklist (A-Level)

- [x] Full dynamic torque computation — Recursive Newton-Euler
- [x] Defined optimization objective — mixed torque + acceleration
- [x] Constrained SLSQP optimization — converges in ~100s, all constraints satisfied
- [x] Comparative plots — baseline vs optimized (torques, velocities, EE acceleration, EE path)
- [x] Constraint satisfaction verified — position, velocity, torque limits all satisfied
- [x] ROS2 Gazebo + RViz simulation with trajectory execution
- [x] Real-time metrics monitoring and CSV logging
- [x] Smoothed real-time qddot estimate (3-sample sliding window FD in metrics node)
