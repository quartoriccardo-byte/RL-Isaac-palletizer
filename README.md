# RL-Isaac-Palletizer

Reinforcement Learning for robotic palletization using Isaac Lab and RSL-RL.

## Task Objective

The agent must **load ALL possible boxes onto the pallet** while respecting hard physical constraints:

- **Maximum Stack Height**: Placements that would exceed `max_stack_height` are masked as invalid actions
- **Maximum Payload**: Episode terminates if prospective on-pallet mass exceeds `max_payload_kg`
- **Physical Stability**: Boxes must not fall or drift excessively after placement

The **buffer** is a **temporary staging area** only—it allows the agent to defer placement decisions strategically, but all boxes must ultimately be placed on the pallet to maximize reward.

---

## Architecture

**Pipeline:** Isaac Lab (DirectRLEnv) + RSL-RL PPO + MultiDiscrete Actions + Warp heightmaps

| Component     | Implementation                                                       |
| ------------- | -------------------------------------------------------------------- |
| Environment   | `pallet_rl.envs.pallet_task.PalletTask` (Isaac Lab `DirectRLEnv`)    |
| Policy        | `pallet_rl.models.rsl_rl_wrapper.PalletizerActorCritic` (MultiDiscrete) |
| Algorithm     | RSL-RL `OnPolicyRunner` with PPO                                     |
| Observations  | Heightmap (Warp GPU kernel) + buffer + box dims + mass/payload + constraints |
| Actions       | MultiDiscrete: [Operation(3), Slot(10), X(16), Y(24), Rotation(2)]  |

### Observation Space (38491-dim)

| Component | Size | Description |
|-----------|------|-------------|
| Heightmap | 38400 | Normalized heights (160×240 grid, Warp GPU kernel) |
| Buffer State | 60 | 10 slots × 6 features [L, W, H, ID, Age, Mass] |
| Box Dimensions | 3 | Current box L, W, H in meters |
| Payload Norm | 1 | `payload_kg / max_payload_kg` |
| Box Mass Norm | 1 | `current_box_mass / max_box_mass` |
| Max Payload Norm | 1 | Normalized constraint (for future domain randomization) |
| Max Stack Height Norm | 1 | Normalized constraint (for future domain randomization) |
| Proprioception | 24 | Robot state (placeholder) |

---

## Constraints & Rewards

### Hard Constraints

| Constraint | Mechanism | Effect |
|------------|-----------|--------|
| **Max Stack Height** | Action masking via `get_action_mask()` | Invalid grid cells masked; attempting blocked → `reward_invalid_height` |
| **Max Payload** | Infeasibility termination | Episode ends if prospective total mass exceeds limit |

### Stability Evaluation

After each PLACE or RETRIEVE, a **settling window** (`settle_steps = 10` physics steps) runs before evaluation:

1. **Physical Collapse Detection**: If any box z < 0.05m → `reward_fall = -25.0` (most severe)
2. **Drift Evaluation**: Compute XY drift and rotation drift from target pose
   - If drift exceeds thresholds → `reward_drift = -3.0`
   - Otherwise → `reward_stable = +1.0`

Drift thresholds:

- `drift_xy_threshold = 0.035m` (3.5cm)
- `drift_rot_threshold = 7.0°`

### Reward Hierarchy

Rewards are tuned to ensure correct learning priority:

| Event | Reward | Rationale |
|-------|--------|-----------|
| Physical collapse (box falls) | **-25.0** | Most severe—safety-critical |
| Infeasible termination | **-4.0** | Moderate—not directly agent's fault |
| Unstable placement (drift) | **-3.0** | Mild—recoverable quality issue |
| Invalid height action | **-2.0** | Small—should be masked anyway |
| Stable successful placement | **+1.0** | Positive reinforcement |

> **Note**: Infeasible termination has a moderate penalty because it depends on box generation, not agent action quality. It should not dominate PPO gradients.

---

## Physical Buffer Semantics

The buffer stores **physical box identities**, not virtual inventory:

- **STORE** parks the last-placed physical box in a holding area, records its ID and mass
- **RETRIEVE** moves that same parked physical box back onto the pallet
- **PLACE** consumes a fresh box from the episode allocation; only PLACE increments `box_idx`

Example: `Place A → Store slot0 → Place B → Retrieve slot0` returns box A (not B or a new box).

When storing/retrieving:

- Mass is tracked in `buffer_state[:, :, 5]`
- STORE removes mass from `payload_kg`
- RETRIEVE adds mass back to `payload_kg`

---

## Termination Conditions

Episodes terminate when:

1. **Physical Collapse**: Any actively placed box falls (z < 0.05m during settling)
2. **Infeasible Mass**: `payload_kg + buffer_mass + remaining_mass > max_payload_kg`
3. **All Boxes Placed**: `box_idx >= max_boxes` (successful completion)
4. **Time Limit**: Episode length exceeded (handled by DirectRLEnv)

---

## Training Metrics

Task KPIs are emitted via `self.extras` and logged to TensorBoard under `metrics/*`:

| Metric | Description |
|--------|-------------|
| `place_success_rate` | Fraction of successful placements |
| `place_failure_rate` | Fraction of failed placements |
| `retrieve_success_rate` | Fraction of successful retrieves |
| `store_accept_rate` | Fraction of valid store attempts |
| `buffer_occupancy` | Average buffer slot utilization |
| `drift_rate` | Fraction of placements with excessive drift |
| `collapse_rate` | Fraction of placements that resulted in collapse |
| `infeasible_rate` | Rate of infeasibility terminations |
| `avg_drift_xy` | Average XY drift in meters |
| `avg_drift_deg` | Average rotation drift in degrees |
| `payload_utilization` | `payload_kg.mean() / max_payload_kg` |

---

## Current Assumptions

- **Pallet is fixed**: No tipping or sliding
- **Box COM approximated**: Rigid body center used for stability
- **No pallet tipping check**: Assumes symmetric loading or robust pallet
- **Constraint values constant**: Currently fixed; observation includes normalized values for future domain randomization support

---

## Installation

**Prerequisites** (on a machine that can run Isaac):

- **Isaac Sim 4.x**: Python 3.10+, Isaac Lab 4.x compatible version
- **Isaac Sim 5.x**: Python 3.11, Isaac Lab 5.x compatible version
- CUDA-capable GPU with recent drivers (tested with RTX 30xx/40xx)

**Install the package** (inside your Isaac Lab Python environment):

```bash
cd RL-Isaac-palletizer   # repo root

# Base install (without RSL-RL training dependencies)
pip install -e .

# Training install (includes RSL-RL from GitHub)
pip install -e ".[train]"

# Quick import check
python -c "import pallet_rl; print('pallet_rl OK')"
```

---

## Training

All commands must be run on a machine with Isaac Lab installed.

```bash
# Full training run (headless, 4096 envs)
python scripts/train.py \
  --headless \
  --num_envs 4096 \
  --device cuda:0 \
  --max_iterations 2000 \
  --log_dir runs/palletizer \
  --experiment_name palletizer_ppo
```

The script:

- Launches Isaac Lab via `isaaclab.app.AppLauncher`
- Instantiates `PalletTask` with `PalletTaskCfg` (DirectRLEnv)
- Wraps it with `RslRlVecEnvWrapper` for RSL-RL
- Uses `PalletizerActorCritic` for MultiDiscrete action distribution

---

## Evaluation

```bash
python scripts/eval.py \
  --headless \
  --num_envs 128 \
  --device cuda:0 \
  --checkpoint path/to/checkpoint.pt
```

---

## Testing

```bash
# Constraint logic tests (no Isaac required)
python tests/test_constraints.py

# Run all tests
python -m pytest tests/ -v
```

---

## Configuration Summary

Key parameters in `PalletTaskCfg`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_stack_height` | 1.8m | Maximum allowed stack height |
| `max_payload_kg` | 500.0 | Maximum on-pallet mass |
| `base_box_mass_kg` | 5.0 | Base mass per box |
| `box_mass_variance` | 2.0 | ± variance for randomization |
| `settle_steps` | 10 | Physics steps for settling window |
| `drift_xy_threshold` | 0.035m | Max allowed XY drift |
| `drift_rot_threshold` | 7.0° | Max allowed rotation drift |
| `reward_fall` | -25.0 | Physical collapse penalty |
| `reward_infeasible` | -4.0 | Infeasibility termination penalty |
| `reward_drift` | -3.0 | Drift penalty |
| `reward_invalid_height` | -2.0 | Height violation penalty |
| `reward_stable` | +1.0 | Stable placement bonus |

---

## Project Structure

```text
pallet_rl/
├── envs/
│   └── pallet_task.py        # Isaac Lab DirectRLEnv palletizing task
├── models/
│   └── rsl_rl_wrapper.py     # PalletizerActorCritic (MultiDiscrete)
├── utils/
│   ├── heightmap_rasterizer.py  # Warp-based GPU heightmap kernel
│   └── quaternions.py           # Isaac↔Warp quaternion conversions
├── configs/
│   └── rsl_rl_config.yaml    # RSL-RL PPO config
scripts/
├── train.py                  # Training entrypoint
└── eval.py                   # Evaluation entrypoint
tests/
├── test_constraints.py       # Constraint logic tests
├── test_bugs.py              # Utility function tests
└── test_quaternions.py       # Quaternion helper tests
```

---

## DirectRLEnv Lifecycle

`PalletTask` respects the Isaac Lab `DirectRLEnv` step lifecycle:

```text
DirectRLEnv.step(action)
  │
  ├─ _pre_physics_step(action)
  ├─ _apply_action(action)         ← Height validation, box placement, buffer logic
  │                                    Payload updates, settling window armed
  ├─ Physics stepping              ← cfg.decimation × sim.step()
  ├─ _post_physics_step()
  ├─ _get_observations()           ← Heightmap, buffer, mass/payload, constraints
  ├─ _get_rewards()                ← Settling evaluation, KPI logging
  ├─ _get_dones()                  ← Termination checks (collapse, infeasible)
  └─ Reset handling
```

The only **supported** RL pipeline is:
`PalletTask` (DirectRLEnv) → `RslRlVecEnvWrapper` → `PalletizerActorCritic` → `OnPolicyRunner`
