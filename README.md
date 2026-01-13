# RL-Isaac-Palletizer

Reinforcement Learning for robotic palletization using Isaac Lab and RSL-RL.

## Architecture

**Chosen Pipeline:** Isaac Lab + RSL-RL PPO + MultiDiscrete Actions

| Component | Implementation |
|:----------|:---------------|
| Environment | `pallet_rl.envs.pallet_task.PalletTask` (Isaac Lab DirectRLEnv) |
| Policy | `pallet_rl.models.rsl_rl_wrapper.PalletizerActorCritic` |
| Algorithm | RSL-RL `OnPolicyRunner` with PPO |
| Actions | MultiDiscrete: [Operation(3), Slot(10), X(16), Y(24), Rotation(2)] |

## Installation

```bash
# Clone and install
cd rl-isaac-palletizer
pip install -e .

# Verify
python -c "import pallet_rl; print('OK')"
```

## Training

```bash
# Inside Isaac Lab environment
python scripts/train.py --headless --config pallet_rl/configs/base.yaml
```

## Evaluation

```bash
python scripts/eval.py --checkpoint runs/rsl_rl_palletizer/model.pt
```

## Known Isaac Lab / RSL-RL Constraints

1. **AppLauncher must be first**: Import `AppLauncher` and call it before any other Isaac imports
2. **Wrapper import paths change**: Isaac Lab ~1.0 uses `omni.isaac.lab_tasks.utils.wrappers.rsl_rl`
3. **MultiDiscrete actions**: RSL-RL expects continuous; we override ActorCritic to handle discrete

## Fixed Critical Bugs (v0.1.0)

| Bug | File | Fix |
|:----|:-----|:----|
| Square grid assumption | `algo/utils.py` | `decode_action` now uses `width * height` |
| Weight destruction | `models/actor_critic.py` | `action_mean` no longer calls `fill_(0)` |
| Invalid super() | `models/rsl_rl_wrapper.py` | Uses `nn.Module.__init__(self)` |
| Mask shape mismatch | `models/policy_heads.py` | Added shape assertion |
| Tensor type mismatch | `scripts/train.py` | `terminated\|truncated` normalized to tensors |

## Runtime Validation Checklist

Before running on a new machine:

- [ ] Isaac Lab installed and `isaacsim` command works
- [ ] `pip install -e .` succeeds
- [ ] `python -c "import pallet_rl"` succeeds
- [ ] `python tests/test_imports.py` passes
- [ ] `python tests/test_bugs.py` passes

## Project Structure

```
pallet_rl/
├── envs/           # Isaac Lab environments
├── models/         # Neural network architectures
├── algo/           # Utility functions (decode_action, etc.)
├── configs/        # YAML configurations
├── utils/          # Warp heightmap rasterizer
└── legacy/         # Archived code (custom PPO, storage)
scripts/
├── train.py        # Training entrypoint
└── eval.py         # Evaluation entrypoint
tests/
├── test_imports.py
└── test_bugs.py
```
