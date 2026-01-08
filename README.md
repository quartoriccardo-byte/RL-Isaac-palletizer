# PalletRL: Isaac Lab Palletizing Agent

This repository contains a Reinforcement Learning implementation for robotic palletizing using NVIDIA Isaac Lab (Isaac Sim) and PPO.

## Prerequisites

- **OS**: Linux (tested on Ubuntu 20.04/22.04) or Windows 10/11
- **GPU**: NVIDIA RTX GPU with drivers supporting CUDA 11.x/12.x
- **Software**:
  - [NVIDIA Isaac Sim 4.0+](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/index.html)
  - [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
  - Python 3.10 (bundled with Isaac Sim) or equivalent environment

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/quartoriccardo-byte/RL-Isaac-palletizer.git
   cd RL-Isaac-palletizer
   ```

2. **Install dependencies**:
   Ensure you are in the python environment where Isaac Lab is installed.

   ```bash
   pip install -e .
   ```

   *Note: Standard dependencies like `torch` are assumed to be present from Isaac Lab installation.*

## Usage

### Training

To launch the PPO training loop:

```bash
# Run with Isaac Lab (Simulated Env)
python src/pallet_rl/train.py --config src/pallet_rl/configs/base.yaml --headless

# Run in Headless mode (no GUI)
python src/pallet_rl/train.py --config src/pallet_rl/configs/base.yaml --headless
```

To run with the GUI (for debugging, slower):

```bash
python src/pallet_rl/train.py --config src/pallet_rl/configs/base.yaml
```

**Note**: If Isaac Lab is not found, the script falls back to a dummy environment (for testing logic).

### Visualization / Evaluation

To evaluate a trained checkpoint:

```bash
python src/pallet_rl/eval.py --checkpoint runs/prod_run/ckpt_500.pt
```

This runs the policy in inference mode. If running with Isaac Lab, it will open the simulator window to visualize the agent's actions.

## Project Structure

- `algo/`: PPO implementation and RolloutBuffer.
- `configs/`: Hyperparameter configurations (`base.yaml`).
- `envs/`: Environment logic.
  - `isaaclab_task.py`: Main Isaac Lab integration (PhysX, Scene, Rewards).
  - `heightmap_channels.py`: Logic for converting heightmaps to observation tensors.
- `models/`: Neural Network Architecture.
  - `encoder2d.py`: ResNet-like 2D Encoder.
  - `unet2d.py`: U-Net for auxiliary tasks/masking (if used).
  - `policy_heads.py`: `SpatialPolicyHead` with action masking.
- `train.py`: Main training entry point.
- `eval.py`: Evaluation entry point.
