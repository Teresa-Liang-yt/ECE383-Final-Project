# ECE383 Final Project — Dynamic Trajectory Optimization for Bartender Robot Arm

Kinova Gen3 Lite 6-DOF arm simulation with mixing-motion trajectory optimization.

**Objective:** Find joint trajectories that maximize end-effector acceleration along a target shake axis (mixing effectiveness) while minimizing joint torques (mechanical safety), subject to joint position, velocity, and torque limits.

---

## Repository Structure

```
ECE383-Final-Project/
├── bartender_arm/              # Core Python library (no ROS2 deps — runs on MacBook)
│   ├── kinematics.py           # Forward kinematics + geometric Jacobian
│   ├── dynamics.py             # Recursive Newton-Euler: τ = M(q)q̈ + C(q,q̇)q̇ + G(q)
│   ├── trajectory.py           # Fourier-series trajectory parameterization
│   ├── optimizer.py            # DE + SLSQP constrained optimization
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

# Recommended: global optimizer (Differential Evolution + SLSQP) — avoids local minima
python3 scripts/run_optimization.py --mode amplitude --global
python3 scripts/run_optimization.py --mode frequency --global

# Fast local-only (multi-start SLSQP, ~100s — good for quick iteration)
python3 scripts/run_optimization.py --mode amplitude
python3 scripts/run_optimization.py --mode frequency
```

**Output files (written to current directory):**
- `comparison_plots_<mode>.png` — 4-panel figure (torques, velocities, EE acceleration, EE path projected onto shake axis)
- `baseline_trajectory.csv` — naive baseline trajectory + torques
- `optimized_trajectory_<mode>.csv` — optimized trajectory + torques
- `<mode>_coeffs.npy` — optimal Fourier coefficients (used by the trajectory publisher node)

**Additional deliverable files (pre-generated, committed to repo):**
- `results_summary_tables.png` — A4-ratio summary of all results metrics, per-joint constraints, and trajectory characteristics
- `results_summary_tables.xlsx` — Same tables in formatted Excel (A4 print-ready, color-coded by mode)

**Expected results (verified with `--global`, corrected RNE dynamics, directional objective):**

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| **Amplitude mode (0.25 Hz)** | | | |
| Peak EE accel (m/s²) | 0.19 | 4.91 | +2553% |
| Mean EE accel (m/s²) | 0.12 | 3.42 | +2701% |
| RMS torque (Nm) | 6.44 | 7.31 | +13.5% |
| Peak joint vel (rad/s) | 0.28 | 1.39 | within limit |
| Y-axis accel fraction | — | 49% | — |
| **Frequency mode (1.0 Hz)** | | | |
| Peak EE accel (m/s²) | 2.95 | 11.24 | +280% |
| Mean EE accel (m/s²) | 1.96 | 7.14 | +265% |
| RMS torque (Nm) | 7.38 | 9.26 | +25.5% |
| Peak joint vel (rad/s) | 1.13 | 1.30 | within limit |
| Y-axis accel fraction | — | 80% | — |

All constraints satisfied. Tightest margins: position >0.82 rad, torque >1.65 Nm (frequency J1), velocity >0.001 rad/s (amplitude, velocity-bound) / >0.098 rad/s (frequency, torque-bound).

### Options

```bash
python3 scripts/run_optimization.py --mode amplitude --no-optimize   # baseline only (fast)
python3 scripts/run_optimization.py --mode amplitude --output-dir ./results/
python3 scripts/run_optimization.py --mode amplitude --load-coeffs amplitude_coeffs.npy  # re-evaluate saved result
```

**Runtime:** ~100 seconds (multi-start SLSQP) or ~9–10 hours (global DE + SLSQP, recommended for final coefficients). Run with `caffeinate -i` on macOS to prevent sleep.

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

The first pull may fail with a "dubious ownership" error because the mounted volume is owned by a different user than the container's git. Fix it once:

```bash
git config --global --add safe.directory /root/workspaces/ECE383-Final-Project
```

Then pull normally:

```bash
cd ~/workspaces/ECE383-Final-Project
git pull
```

### Optimized coefficients

`amplitude_coeffs.npy` and `frequency_coeffs.npy` are force-tracked in git (added with `git add -f` despite the `.gitignore` entry), so they **will** sync automatically via `git pull`. No separate file transfer is needed.

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

#### 1. Fix xacro gripper parameter mismatch

The Docker image has a version mismatch: `kortex_bringup` passes `gripper:=gen3_lite_2f` to xacro, but `gen3_lite_macro.xacro` does not declare that parameter. Patch it once:

```bash
XACRO=/opt/kortex_ws/install/kortex_description/share/kortex_description/arms/gen3_lite/6dof/urdf/gen3_lite_macro.xacro

sed -i 's|<xacro:macro name="load_arm"|<xacro:arg name="gripper" default="gen3_lite_2f"/>\n  <xacro:macro name="load_arm"|' $XACRO
```

> This edits a file inside the container. If you restart with `--rm`, re-run this command next session.

#### 2. Create the colcon symlink

Colcon requires source packages to live under a `src/` directory:

```bash
mkdir -p ~/workspaces/src
ln -sfn ~/workspaces/ECE383-Final-Project ~/workspaces/src/bartender_arm
ls -la ~/workspaces/src/bartender_arm   # verify: should point to ECE383-Final-Project
```

#### 3. Install Python dependencies

```bash
pip3 install scipy numpy matplotlib pyyaml
```

#### 4. Build the package

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

#### 5. Copy optimized coefficients into the install directory

This step must happen **after** `colcon build` (the build creates the install directory). The trajectory publisher looks for coefficients in the package share directory:

```bash
cp ~/workspaces/ECE383-Final-Project/amplitude_coeffs.npy \
   ~/workspaces/install/bartender_arm/share/bartender_arm/amplitude_coeffs.npy
cp ~/workspaces/ECE383-Final-Project/frequency_coeffs.npy \
   ~/workspaces/install/bartender_arm/share/bartender_arm/frequency_coeffs.npy
```

> If you rebuild with `colcon build`, re-run these `cp` commands — colcon refreshes the install directory and overwrites it.

#### After every `git pull`

If you pull new code, rebuild and re-copy the coefficients:

```bash
cd ~/workspaces
colcon build --packages-select bartender_arm
source install/setup.bash
cp ~/workspaces/ECE383-Final-Project/amplitude_coeffs.npy \
   ~/workspaces/install/bartender_arm/share/bartender_arm/amplitude_coeffs.npy
cp ~/workspaces/ECE383-Final-Project/frequency_coeffs.npy \
   ~/workspaces/install/bartender_arm/share/bartender_arm/frequency_coeffs.npy
```

---

## Part 4 — Running the Simulation

> All commands below run **inside the Docker container**.

Source ROS2 and the workspace overlay in **every new terminal pane** before running anything (skip if you added them to `~/.bashrc` already):

```bash
source /opt/ros/jazzy/setup.bash
source ~/workspaces/install/setup.bash
```

### Trajectory modes

| Mode | Frequency | Amplitude | Description |
|------|-----------|-----------|-------------|
| `amplitude` | 0.25 Hz | large (up to 0.25 rad) | Slow sweeping motion — maximizes joint displacement |
| `frequency` | 1.0 Hz | small (up to 0.06 rad) | Fast tight mixing — maximizes oscillation rate |
| `baseline` | 0.25 Hz | — | Single-joint sinusoid reference (no optimization) |

Each mode has its own pre-optimized coefficients file (`amplitude_coeffs.npy` / `frequency_coeffs.npy`). If the file is missing, the publisher falls back to baseline and logs a warning.

### Global optimum vs local minimum results

Two sets of results are committed for report comparison:

| Version | Config | Arm posture | Coeff files | Box bound |
|---------|--------|-------------|-------------|-----------|
| **Global** (DE + SLSQP) | `robot_params.yaml` | Retracted — `[0, 0.52, -1.57, 0, 1.57, 0]` | `amplitude_coeffs.npy` / `frequency_coeffs.npy` | 0.25 / 0.06 rad |
| **Local** (multi-start SLSQP) | `robot_params_local.yaml` | Extended upward — `[0, 0.35, -0.52, 0, 0.87, 0]` | `amplitude_coeffs_local.npy` / `frequency_coeffs_local.npy` | 0.10 / 0.026 rad |

To run the **local minimum** version on the VM, pass both the alternate config and coefficients to the publisher:

```bash
# Copy local coefficients to install directory first
cp ~/workspaces/ECE383-Final-Project/amplitude_coeffs_local.npy \
   ~/workspaces/install/bartender_arm/share/bartender_arm/amplitude_coeffs_local.npy
cp ~/workspaces/ECE383-Final-Project/frequency_coeffs_local.npy \
   ~/workspaces/install/bartender_arm/share/bartender_arm/frequency_coeffs_local.npy
cp ~/workspaces/ECE383-Final-Project/config/robot_params_local.yaml \
   ~/workspaces/install/bartender_arm/share/bartender_arm/config/robot_params_local.yaml

# Run amplitude local-min (arm extended upward posture)
ros2 run bartender_arm trajectory_publisher_node --ros-args \
  -p trajectory_mode:=amplitude \
  -p use_sim_time:=true \
  -p config_path:=$(ros2 pkg prefix bartender_arm)/share/bartender_arm/config/robot_params_local.yaml \
  -p coeffs_path:=$(ros2 pkg prefix bartender_arm)/share/bartender_arm/amplitude_coeffs_local.npy

# Run frequency local-min
ros2 run bartender_arm trajectory_publisher_node --ros-args \
  -p trajectory_mode:=frequency \
  -p use_sim_time:=true \
  -p config_path:=$(ros2 pkg prefix bartender_arm)/share/bartender_arm/config/robot_params_local.yaml \
  -p coeffs_path:=$(ros2 pkg prefix bartender_arm)/share/bartender_arm/frequency_coeffs_local.npy
```

### Option A: One-command launch (recommended)

```bash
ros2 launch bartender_arm simulation.launch.py                           # amplitude mode (default)
ros2 launch bartender_arm simulation.launch.py trajectory_mode:=frequency
ros2 launch bartender_arm simulation.launch.py trajectory_mode:=baseline
```

This automatically starts Gazebo, then RViz after a 5-second delay, then the trajectory publisher after an 8-second delay. The metrics subscriber starts immediately.

### Option B: Manual launch (for debugging — use separate Terminator panes)

**Pane 1 — Gazebo + ros2_control:**
```bash
ros2 launch kortex_bringup kortex_sim_control.launch.py \
  robot_type:=gen3_lite dof:=6 gripper:=gen3_lite_2f robot_name:=gen3_lite \
  sim_gazebo:=true launch_rviz:=false use_sim_time:=true \
  robot_controller:=joint_trajectory_controller
```

**Pane 2 — RViz (wait ~5s for Gazebo to finish loading):**
```bash
source /opt/ros/jazzy/setup.bash && source ~/workspaces/install/setup.bash
ros2 run rviz2 rviz2 \
  -d ~/workspaces/install/bartender_arm/share/bartender_arm/rviz/bartender_arm.rviz
```

> The `Stereo is NOT SUPPORTED` and OpenGL messages are normal — not errors.

**Pane 3 — Metrics subscriber:**
```bash
source /opt/ros/jazzy/setup.bash && source ~/workspaces/install/setup.bash
ros2 run bartender_arm metrics_subscriber_node \
  --ros-args -p use_sim_time:=true
```

**Pane 4 — Trajectory publisher (wait ~8s for the controller to be ready):**
```bash
source /opt/ros/jazzy/setup.bash && source ~/workspaces/install/setup.bash

# Amplitude mode — large slow sweep (0.25 Hz)
ros2 run bartender_arm trajectory_publisher_node \
  --ros-args -p trajectory_mode:=amplitude -p use_sim_time:=true

# Frequency mode — fast tight mixing (1.0 Hz)
ros2 run bartender_arm trajectory_publisher_node \
  --ros-args -p trajectory_mode:=frequency -p use_sim_time:=true

# Baseline — unoptimized single-joint sinusoid reference
ros2 run bartender_arm trajectory_publisher_node \
  --ros-args -p trajectory_mode:=baseline -p use_sim_time:=true
```

> **`use_sim_time:=true` is required.** Gazebo runs on simulation time (starts at t=0). Without this flag the trajectory publisher stamps messages with wall-clock time (~Unix epoch), the controller sees waypoints scheduled billions of seconds in the future, and silently ignores them — the arm will not move.

> Confirm coefficients loaded: the log should say `Mode=amplitude | f=0.25 Hz | loaded coefficients from .../amplitude_coeffs.npy`. If it says `Mode=baseline` instead, the `.npy` files were not copied to the install directory — re-run the `cp` commands from Step 3f.

### Monitor metrics in real-time

```bash
ros2 topic echo /bartender/metrics_summary    # [peak_torque, rms_torque, mean_ee_accel]
ros2 topic echo /bartender/joint_torques
ros2 topic echo /bartender/ee_acceleration
cat /tmp/bartender_metrics.csv                # full log (inside container)
```

> Real-time `qddot` is estimated via finite difference over a 3-sample sliding window, spanning two time steps to reduce single-sample derivative noise. The offline optimization uses analytic derivatives and is unaffected.

---

## Part 5 — Running on the Real Robot Arm

> All commands below run **inside the Docker container** with the physical Kinova Gen3 Lite connected via USB/Ethernet.

Source the workspace first (skip if already in `~/.bashrc`):

```bash
source /opt/ros/jazzy/setup.bash
source ~/workspaces/install/setup.bash
```

### Step 1 — Start the Kinova driver (starts RViz automatically)

```bash
ros2 launch kortex_bringup gen3_lite.launch.py
```

> **Note:** The exact launch file name may vary — if the above doesn't work, check available launch files with `ros2 launch kortex_bringup <TAB>`. The driver brings up `ros2_control`, connects to the arm over the network, and opens RViz.

### Step 2 — Start the metrics subscriber (new Terminator pane)

```bash
ros2 run bartender_arm metrics_subscriber_node
```

> Unlike simulation, omit `use_sim_time` — the real robot uses wall clock time.

### Step 3 — Start the trajectory publisher (new Terminator pane)

```bash
# amplitude mode — large slow sweep (default)
ros2 run bartender_arm trajectory_publisher_node

# or frequency mode — fast tight mixing
ros2 run bartender_arm trajectory_publisher_node \
  --ros-args -p trajectory_mode:=frequency
```

The arm will begin executing the trajectory immediately. The publisher loops automatically (re-sends the trajectory every cycle duration).

### Monitor

```bash
ros2 topic echo /bartender/joint_torques
ros2 topic echo /bartender/ee_acceleration
ros2 topic echo /bartender/metrics_summary    # [peak_torque, rms_torque, mean_ee_accel]
cat /tmp/bartender_metrics.csv
```

> **Safety:** The trajectory publisher sends waypoints to `joint_trajectory_controller`. The arm will move — make sure the workspace is clear before starting Step 3.

---

## Engineering Design

### Trajectory Parameterization

Fourier series over 6 joints with K=3 harmonics:

```
q_i(t) = q0_i + Σ_{k=1}^{3} [ a_{i,k}·sin(k·2πf·t) + b_{i,k}·cos(k·2πf·t) ]
```

36 optimization variables total (2 coefficients × 6 joints × 3 harmonics). Analytic first and second derivatives — no finite differences during optimization.

The trajectory is periodic by construction (period T = 1/f), which prevents drift and produces repeatable cyclic mixing motions.

### Dynamics Model

Full Recursive Newton-Euler (RNE) using inertial parameters from `kortex_description` URDF:

```
τ = M(q)q̈ + C(q,q̇)q̇ + G(q)
```

Verified: `M(q)@qddot + C + G == rne(q,qdot,qddot)` to machine precision (< 1e-15 error).

### Optimization

**Objective** (minimize torque, maximize directional end-effector acceleration):

```
J(x) = (1/N) Σ_t [ 1.0·||τ(t)||²  −  50·(a_ee(t)·n̂)²  +  1.0·||p_ee_perp(t)||² ]
```

where `n̂ = [0, 1, 0]` (world Y axis — along the bottle/tool axis), and `p_ee_perp` is EE position drift perpendicular to the shake axis. The directional acceleration term `(a_ee·n̂)²` rewards only back-and-forth motion along the shake axis; purely circular orbits generate no reward. The drift penalty keeps the EE near its center position.

**Constraints:**

```
q_min[i]  ≤ q_i(t)                       ≤ q_max[i]       (joint position limits, sampled)
Σ_k kω√(a_ik²+b_ik²)                     ≤ 0.95·v_max[i]  (velocity limit, analytic upper bound)
|τ_i(t)|                                  ≤ 0.95·τ_max[i]  (torque limits, sampled, 5% margin)
Σ x²                                      ≥ 0.001           (non-trivial motion)
```

The velocity constraint uses the triangle-inequality bound on `|qdot_i(t)|` for all `t`, not a sampled check. This is necessary because the 3rd harmonic at f=1 Hz oscillates at 3 Hz — with only 60 optimization time points over 6 seconds (~3.3 samples/cycle), sampled velocity peaks are missed. The analytic bound is exact regardless of the time grid. The 5% safety margin on torques keeps SLSQP's linesearch away from constraint boundaries during refinement.

**Solver — two-phase global optimization (`--global`):**

1. **Phase 1 — Differential Evolution:** Population of 180 individuals (`5 × 36 vars`) evolved for up to 200 generations with the same constraints. Escapes local minima without requiring a good initial guess.
2. **Phase 2 — SLSQP refinement:** Polishes the DE solution to tighter convergence. If SLSQP diverges but the stopped point is still physically feasible, it is kept (it is typically better than the DE solution). If SLSQP leaves the feasible region, the DE solution is used instead.

Fast local-only mode (no `--global`) runs 16-start multi-start SLSQP in ~100 seconds, suitable for iteration but more likely to find local optima.

---

## Grading Checklist (A-Level)

- [x] Full dynamic torque computation — Recursive Newton-Euler, verified to machine precision
- [x] Defined optimization objective — directional EE acceleration along shake axis + torque minimization + drift penalty
- [x] Constrained global optimization — Differential Evolution + SLSQP, all physical constraints satisfied
- [x] Comparative plots — baseline vs optimized (torques, velocities, EE acceleration, EE path projected onto shake axis) — `comparison_plots_amplitude.png`, `comparison_plots_frequency.png`
- [x] Constraint satisfaction verified — position, velocity, torque limits all satisfied with explicit margins (see `results_summary_tables.png` / `.xlsx`)
- [x] ROS2 Gazebo + RViz simulation with trajectory execution
- [x] Real-time metrics monitoring and CSV logging
- [x] Smoothed real-time qddot estimate (3-sample sliding window FD in metrics node)
