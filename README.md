# PalletRL: Isaac Lab Palletizing Agent

This repository contains a Reinforcement Learning implementation for robotic palletizing using NVIDIA Isaac Lab (Isaac Sim) and PPO.

## Quick Start

### Installation

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

### Training

To launch the training loop in headless mode (recommended for training):

```bash
python src/pallet_rl/train.py --headless
```

### Architecture

The agent uses an **Actor-Critic** architecture:

1. **Encoder**: A 3-layer CNN (`Encoder2D`) that processes the heightmap observation into latent features.
2. **Spatial Policy Head**: A convolutional head that outputs a probability map `(Batch, Rotations, H, W)` representing the logits for placing a box at a specific `(x, y)` location with a specific rotation.
    * **Action Masking**: Invalid actions (e.g., overhangs) are masked out with `-1e8` to ensure the agent only selects valid placements.

## Project Structure

* `algo/`: PPO implementation (`ppo.py`).
* `configs/`: Hyperparameter configurations (`base.yaml`).
* `envs/`: Environment logic.
* `models/`: Neural Network Architecture (`encoder2d.py`, `policy_heads.py`).
* `train.py`: Main training script.
