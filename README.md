# PalletRL

A Reinforcement Learning project for palletization tasks using NVIDIA Isaac Lab.

## Overview

This project implements a PPO agent that learns to stack boxes on a pallet. It uses a custom Actor-Critic architecture with:

- **Encoder2D**: Extracts features from the heightmap.
- **UNet2D**: Generates masks/attention.
- **PolicyHeads**: Outputs placement actions (Pick, Yaw, X, Y).

## Installation

1. Install PyTorch.
2. Install NVIDIA Isaac Lab (for full simulation).
3. Install dependencies:

    ```bash
    pip install numpy torch pyyaml scipy
    ```

## Usage

### Local Testing (No Simulator)

Run the training loop with a dummy environment:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m pallet_rl.train --config test_config.yaml
```

### Full Simulation

Run with Isaac Lab:

```bash
export USE_ISAACLAB=1
python -m pallet_rl.train --config configs/base.yaml
```

## Structure

- `algo/`: PPO implementation and storage.
- `envs/`: Environment logic (`IsaacLabVecEnv`, `DummyVecEnv`).
- `models/`: Neural network architecture.
- `train.py`: Main training loop.
