# RL-Isaac-Palletizer

Reinforcement Learning for robotic palletization using Isaac Lab and RSL-RL.

## Architecture

**Chosen Pipeline:** Isaac Lab (DirectRLEnv) + RSL-RL PPO + MultiDiscrete Actions + Warp heightmaps

| Component     | Implementation                                                       |
| ------------- | -------------------------------------------------------------------- |
| Environment   | `pallet_rl.envs.pallet_task.PalletTask` (Isaac Lab `DirectRLEnv`)    |
| Policy        | `pallet_rl.models.rsl_rl_wrapper.PalletizerActorCritic` (MultiDiscrete) |
| Algorithm     | RSL-RL `OnPolicyRunner` with PPO                                     |
| Observations  | Heightmap (Warp GPU kernel) + buffer + box dims + proprio           |
| Actions       | MultiDiscrete: [Operation(3), Slot(10), X(16), Y(24), Rotation(2)]  |

For a detailed snapshot of the current layout and data flow, see `docs/architecture_as_is.md`.

## Installation

- **Prerequisites (on a machine that can run Isaac)**:
  - **Isaac Sim 4.x**: Python 3.10+, Isaac Lab 4.x compatible version.
  - **Isaac Sim 5.x**: Python 3.11, Isaac Lab 5.x compatible version.
  - CUDA‑capable GPU with recent drivers (tested with RTX 30xx/40xx).

- **Install the package** (inside your Isaac Lab Python environment):

```bash
cd RL-Isaac-palletizer   # repo root
pip install -e .

# Quick import check
python -c "import pallet_rl; print('pallet_rl OK')"
```

- **Install RSL-RL** (required dependency):

RSL-RL is listed in `pyproject.toml`, but installation may vary by environment. Choose one:

**Option 1 (recommended, pinned version):**

```bash
pip install git+https://github.com/leggedrobotics/rsl_rl.git@<commit-hash>
```

Replace `<commit-hash>` with a specific commit hash for reproducibility (e.g., `@abc123def456`).

**Option 2 (editable from local clone):**

```bash
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl
pip install -e .
```

**Option 3 (Isaac Lab environment):**
Some Isaac Lab environments may already include `rsl_rl`. Verify with:

```bash
python -c "import rsl_rl; print('rsl_rl OK')"
```

If import fails, use Option 1 or 2 above. The scripts will show a clear error message if `rsl_rl` is missing.

## Training (Canonical)

All commands in this section must be run on a machine that can launch Isaac Lab. Do **not** run them on this repository’s editing machine if Isaac is unavailable.

```bash
# Example: headless training with 4096 envs on GPU 0
python scripts/train.py \
  --headless \
  --num_envs 4096 \
  --device cuda:0 \
  --max_iterations 2000 \
  --log_dir runs/palletizer \
  --experiment_name palletizer_ppo
```

The script:

- Launches Isaac Lab via `isaaclab.app.AppLauncher` (inside `main()` to avoid import-time side effects).
- Instantiates `PalletTask` with `PalletTaskCfg` (DirectRLEnv).
- Wraps it with `RslRlVecEnvWrapper` for RSL‑RL.
- Loads RSL‑RL hyper‑parameters from `pallet_rl/configs/rsl_rl_config.yaml` (then applies CLI overrides).
- Monkey‑patches `rsl_rl.modules.ActorCritic` to use `PalletizerActorCritic` (MultiDiscrete policy).

**Logs and checkpoints:**

- RSL‑RL writes logs and checkpoints under `--log_dir` and `--experiment_name`.
- With the default flags, you will see runs in `runs/palletizer/palletizer_ppo/*`
  including TensorBoard `events.*` files and `.pt` checkpoints.

## Evaluation (Canonical)

After training, evaluate a checkpoint using the same env and policy:

```bash
python scripts/eval.py \
  --headless \
  --num_envs 128 \
  --device cuda:0 \
  --checkpoint path/to/checkpoint.pt
```

This script:

- Launches Isaac Lab.
- Builds the same `PalletTask` + `RslRlVecEnvWrapper` stack.
- Injects `PalletizerActorCritic` into RSL‑RL.
- Loads the checkpoint and runs rollouts using `act_inference` (deterministic argmax actions).

## Isaac Lab / RSL-RL Constraints

- **AppLauncher ordering**: `isaaclab.app.AppLauncher` must be constructed **before** any other Isaac Lab imports that touch simulation.
- **Launcher CLI args**: `train.py` and `eval.py` define a subset of AppLauncher CLI args locally (headless, livestream, video, etc.) to avoid importing Isaac modules at parse time. Add additional launcher flags to `parse_args()` if needed.
- **Wrapper path**: This repo targets the modern import path `isaaclab.envs.wrappers.rsl_rl.RslRlVecEnvWrapper` (not the legacy `omni.isaac.*` namespaces).
- **Action space**: RSL‑RL's PPO is originally continuous; this repo provides a custom `PalletizerActorCritic` that implements a factored MultiDiscrete distribution (per‑dimension `Categorical` with summed log‑prob and entropy).

### DirectRLEnv Lifecycle Compliance

`PalletTask` respects the Isaac Lab `DirectRLEnv` step lifecycle (no custom `step()` override):

```text
DirectRLEnv.step(action)
  │
  ├─ _pre_physics_step(action)
  │
  ├─ _apply_action(action)         ← Task logic: parses action, sets masks,
  │                                    writes box pose to sim, handles buffer,
  │                                    increments box_idx ONLY on Place (op=0)
  │
  ├─ Physics stepping              ← cfg.decimation × sim.step() (default: 50)
  │
  ├─ _post_physics_step()
  │
  ├─ _get_observations()           ← Uses NEW box_idx for next box
  ├─ _get_rewards()                ← Uses box_idx - 1 for placed box evaluation
  ├─ _get_dones()                  ← Uses box_idx - 1 for placed box evaluation
  │
  └─ Reset handling + episode_length_buf updates
```

**Key design decision**: `box_idx` is incremented at the end of `_apply_action()`. This ensures:

- Rewards and termination flags evaluate the **placed** box (`box_idx - 1`).
- Observations show the **next** box to be placed (`box_idx`).

**Physics stepping**: Controlled via `cfg.decimation` (default: 50 physics steps per RL step at `dt=1/60s`).

---

## Complete Command Reference

All commands must be run on a machine with Isaac Lab installed.

### Installation

```bash
# Clone the repository
git clone https://github.com/quartoriccardo-byte/RL-Isaac-palletizer.git
cd RL-Isaac-palletizer

# Install the package (editable mode)
pip install -e .

# Verify installation
python -c "import pallet_rl; print('pallet_rl OK')"

# Install RSL-RL (if not already installed)
pip install git+https://github.com/leggedrobotics/rsl_rl.git
```

### Training

```bash
# Full training run (headless, 4096 envs)
python scripts/train.py \
  --headless \
  --num_envs 4096 \
  --device cuda:0 \
  --max_iterations 2000 \
  --log_dir runs/palletizer \
  --experiment_name palletizer_ppo

# Short smoke test (50 iterations)
python scripts/train.py \
  --headless \
  --num_envs 512 \
  --device cuda:0 \
  --max_iterations 50 \
  --log_dir runs/test_run \
  --experiment_name palletizer_smoke
```

### Evaluation

```bash
# Evaluate a checkpoint
python scripts/eval.py \
  --headless \
  --num_envs 128 \
  --device cuda:0 \
  --checkpoint path/to/checkpoint.pt

# With rendering (remove --headless)
python scripts/eval.py \
  --num_envs 16 \
  --device cuda:0 \
  --checkpoint path/to/checkpoint.pt
```

### Testing (Non-Isaac)

These tests run without Isaac Lab:

```bash
# Import sanity tests
python -m pytest tests/test_imports.py -v

# Static bug tests (utilities, wrappers)
python -m pytest tests/test_bugs.py -v

# Quaternion helper tests
python -m pytest tests/test_quaternions.py -v

# Run all non-Isaac tests
python -m pytest tests/ -v
```

### Git Workflow

```bash
# Check repository status
git status
git fetch --all --prune

# Update to latest main
git checkout main
git pull --ff-only origin main

# After making changes
git add -A
git commit -m "Your commit message"
git push origin main
```

## Validation Pipeline (To Run on an Isaac Machine)

This section describes a complete **validation pipeline** you should run on a machine with Isaac Lab installed. It is **not** executed here.

### 1. Import & Config Sanity

- **Check installation and imports**:

```bash
pip install -e .
python -c "import pallet_rl; print('pallet_rl OK')"
python -m pytest tests/test_imports.py
python -m pytest tests/test_quaternions.py
```

- **Static config inspection**:
  - Open `pallet_rl/configs/rsl_rl_config.yaml` and check that:
    - `env.obs_dim == 38477`.
    - `env.action_dims == [3, 10, 16, 24, 2]`.
    - `runner.policy_class_name == ActorCritic` and `algorithm_class_name == PPO`.

### 2. Environment Smoke Test (No Training)

On an Isaac machine:

```bash
python - << 'EOF'
from isaaclab.app import AppLauncher
from pallet_rl.envs.pallet_task import PalletTask, PalletTaskCfg

class Args: headless=True
app = AppLauncher(Args()).app

cfg = PalletTaskCfg()
cfg.scene.num_envs = 8
env = PalletTask(cfg=cfg, render_mode=None)

obs = env.reset()
print("Reset OK, obs keys:", obs.keys())

import torch
from gymnasium.spaces import MultiDiscrete

assert isinstance(env.action_space, MultiDiscrete)
actions = torch.zeros(env.num_envs, len(env.cfg.action_dims), dtype=torch.long, device=cfg.sim.device)
obs, rew, terminated, truncated, info = env.step(actions)
print("Step OK, reward shape:", rew.shape)

app.close()
EOF
```

Expected:

- Reset and one step complete without exceptions.
- Observation tensors have shape `(num_envs, 38477)` for both `policy` and `critic` keys.

### 3. Quaternion & Heightmap Unit Checks

Run the quaternion helper tests:

```bash
python -m pytest tests/test_quaternions.py
```

On an Isaac+CUDA machine, you can add a small script to sanity‑check Warp rasterization orientation (optional but recommended):

```bash
python - << 'EOF'
import torch
from pallet_rl.utils.heightmap_rasterizer import WarpHeightmapGenerator
from pallet_rl.utils.quaternions import wxyz_to_xyzw

num_envs, max_boxes = 1, 1
gen = WarpHeightmapGenerator(
    device="cuda:0",
    num_envs=num_envs,
    max_boxes=max_boxes,
    grid_res=0.05,
    map_size=(16, 16),
    pallet_dims=(1.0, 1.0),
)

box_pos = torch.tensor([[0.0, 0.0, 0.5]], device="cuda:0")
box_dims = torch.tensor([[0.4, 0.4, 1.0]], device="cuda:0")
q_wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda:0")
box_rot = wxyz_to_xyzw(q_wxyz)
pallet_pos = torch.zeros(1, 3, device="cuda:0")

hm = gen.forward(box_pos, box_rot, box_dims, pallet_pos)
print("Heightmap range:", hm.min().item(), hm.max().item())
EOF
```

You should observe a non‑zero region in the heightmap directly under the box footprint.

### 4. Short Training Run

Run a short PPO training session to verify the full pipeline:

```bash
python scripts/train.py \
  --headless \
  --num_envs 512 \
  --device cuda:0 \
  --max_iterations 50 \
  --log_dir runs/test_run \
  --experiment_name palletizer_smoke
```

Expected artifacts:

- A new directory under `runs/test_run` containing:
  - TensorBoard `events.out.tfevents...` files.
  - RSL‑RL checkpoints (e.g. `model_XXX.pt` or similar, depending on RSL‑RL version).
- Console logs showing PPO iterations, approximate rewards, and KL metrics.

### 5. Evaluation Procedure

Use one of the saved checkpoints:

```bash
CKPT=path/to/checkpoint.pt  # replace with an actual file
python scripts/eval.py \
  --headless \
  --num_envs 128 \
  --device cuda:0 \
  --checkpoint "$CKPT"
```

Observe:

- No crashes when loading the checkpoint.
- Stable rollouts (no obvious numerical explosions, NaNs, etc.).
- If you run with rendering enabled (omit `--headless`), you should see boxes being placed onto the pallet in the viewer.

### 6. Troubleshooting Checklist

- **ImportError / ModuleNotFoundError**:
  - Confirm the correct Python interpreter (Isaac Lab) is used.
  - Re‑run `pip install -e .` and `python tests/test_imports.py`.
- **Warp / CUDA errors**:
  - Ensure `warp-lang` is installed with GPU support.
  - Check that your `CUDA_VISIBLE_DEVICES` and Isaac Lab GPU settings are correct.
- **Quaternion orientation issues (boxes appear rotated incorrectly)**:
  - Verify that `wxyz_to_xyzw` is used at every Isaac→Warp boundary.
  - Re‑run `tests/test_quaternions.py` and the heightmap sanity script.
- **NaNs in training**:
  - Check PPO hyper‑parameters in `pallet_rl/configs/rsl_rl_config.yaml`.
  - Try reducing `learning_rate`, `num_steps_per_env`, or `entropy_coef`.
  - Ensure there are no manual `.nan_to_num` or silent clamping operations hiding numerical problems.

## Project Structure (Current)

```text
pallet_rl/
├── envs/
│   ├── pallet_task.py        # Isaac Lab DirectRLEnv palletizing task
│   └── heightmap_channels.py # Legacy CPU channel computation (not in hot path)
├── models/
│   └── rsl_rl_wrapper.py     # PalletizerActorCritic (MultiDiscrete)
├── algo/
│   └── utils.py              # decode_action, load_config (legacy helper)
├── utils/
│   ├── heightmap_rasterizer.py  # Warp-based GPU heightmap kernel
│   └── quaternions.py           # Isaac↔Warp quaternion conversions
├── configs/
│   ├── rsl_rl_config.yaml    # Canonical RSL-RL PPO config
│   └── base.yaml             # Legacy config (U-Net + masks), kept for reference
legacy/
├── gen_mask_dataset.py       # Archived mask dataset script (U-Net pipeline)
└── profile_sim.py            # Archived profiling placeholder
scripts/
├── train.py                  # Canonical training entrypoint (Isaac Lab + RSL-RL)
├── eval.py                   # Canonical evaluation entrypoint
├── gen_mask_dataset.py       # Thin shim into legacy/gen_mask_dataset.py
└── profile_sim.py            # Thin shim into legacy/profile_sim.py
tests/
├── test_imports.py           # Package import sanity
├── test_bugs.py              # Static tests for utility functions & wrappers
└── test_quaternions.py       # Quaternion helper tests
```

The only **supported** RL pipeline going forward is:
`PalletTask` (DirectRLEnv) → `RslRlVecEnvWrapper` → `PalletizerActorCritic` → `OnPolicyRunner`.
