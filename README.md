# gen3-ros2-env

A tiny, practical ROS 2 (Jazzy) Python environment for Kinova Gen3. It subscribes to joints and RGB topics, shapes images to 224×224 (center-crop or black padding), and exposes a simple `Env` API with `reset()`, `observe()`, and `step()`.

## Features

- 🔌 Plugs into existing ROS 2 bringup; no custom messages or services
- 🖼️ Image shaping utilities (center square crop **or** square with black padding → 224×224)
- 🦾 Two control paths:
  - `twist` → publish `/twist_controller/commands`
  - `joint_trajectory` → publish `/joint_trajectory_controller/joint_trajectory`
- 🧰 Minimal dependencies (NumPy, Pillow, OpenCV) — ROS bits from your distro

## Requirements

- Ubuntu 24.04 (recommended) with **ROS 2 Jazzy** installed
- You have your Kinova Vision node publishing a color stream, e.g. `/camera/color/image_raw`
- Python ≥ 3.10
- System ROS packages provide `rclpy` and `cv_bridge`

## Install

Clone and install Python deps (ROS deps come from apt/your ROS install):

```bash
git clone https://github.com/anashoussaini/gen3-ros2-env.git
cd gen3-ros2-env

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
# optional (editable install):
pip install -e .
