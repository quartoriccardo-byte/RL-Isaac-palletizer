## RL-Isaac-Palletizer – Architecture (As Is)

This document captures the current high‑level architecture of the `RL-Isaac-palletizer` repo as it exists in this workspace, before additional refactors.

### Top‑Level Overview

- **Package**: `pallet_rl`
- **Primary RL stack**: **Isaac Lab (DirectRLEnv) + RSL‑RL (OnPolicyRunner)**
- **Task**: Heightmap‑based palletizing with a discrete, factored (MultiDiscrete) action space.

### Entrypoints

- **Training**: `scripts/train.py`
  - Launches Isaac Lab via `isaaclab.app.AppLauncher`.
  - Instantiates `pallet_rl.envs.pallet_task.PalletTask` with `PalletTaskCfg`.
  - Wraps the env with `isaaclab.envs.wrappers.rsl_rl.RslRlVecEnvWrapper`.
  - Monkey‑patches `rsl_rl.modules.ActorCritic` to `pallet_rl.models.rsl_rl_wrapper.PalletizerActorCritic`.
  - Creates and runs `rsl_rl.runners.OnPolicyRunner` with an in‑code PPO config (currently not loaded from YAML).

- **Evaluation (legacy / mismatched)**: `scripts/eval.py`
  - Uses legacy components (`vec_env_setup`, `Encoder2D`, `UNet2D`, `SpatialPolicyHead`) that do **not** exist in the current package.
  - Uses its own `ActorCritic` unrelated to `PalletizerActorCritic`.
  - Uses a mask‑based spatial policy that is no longer part of the active training pipeline.
  - Conclusion: **structurally inconsistent** with the current DirectRLEnv + RSL‑RL pipeline.

- **Misc / legacy utilities**:
  - `scripts/gen_mask_dataset.py`
    - Depends on `envs.mask_dataset.FIFODataset` (non‑existent) and on CPU numpy for random data.
    - Intended to generate a dataset of state/mask pairs for a former U‑Net policy.
  - `scripts/profile_sim.py`
    - Placeholder; no real integration with Isaac Lab yet.

### Environments

- **Canonical env**: `pallet_rl.envs.pallet_task.PalletTask`
  - Base class: `isaaclab.envs.DirectRLEnv`.
  - Config: `pallet_rl.envs.pallet_task.PalletTaskCfg` (a `@configclass` subclass of `DirectRLEnvCfg`).
  - **Scene / simulation config**:
    - `sim: SimulationCfg` with:
      - `dt = 1/60`
      - `render_interval = 2`
      - `device = "cuda:0"`
    - `scene: InteractiveSceneCfg` with:
      - `num_envs = 4096`
      - `env_spacing = 3.0`
    - `decimation = 10`, `episode_length_s = 60.0`.
  - **Task‑specific config**:
    - Pallet: `pallet_size = (1.2, 0.8)` meters.
    - Heightmap: `map_shape = (160, 240)`, `grid_res = 0.005`, `max_height = 2.0`.
    - Boxes: `max_boxes = 50`.
    - Buffer: `buffer_slots = 10`, `buffer_features = 5` (`[L, W, H, ID, Age]`).
    - Proprio: `robot_state_dim = 24`.
    - Computed obs dim: `num_observations = 38477` (flattened heightmap + buffer + box dims + proprio).
    - Action dims (MultiDiscrete): `(3, 10, 16, 24, 2)` → op, slot, grid x, grid y, rotation.
  - **Core methods**:
    - `_init_state_tensors`:
      - Allocates all per‑env tensors on GPU:
        - `box_dims` `(N, max_boxes, 3)`
        - `buffer_state` `(N, buffer_slots, buffer_features)`
        - index/mask tensors for reward logic (`box_idx`, `active_place_mask`, etc.).
    - `_setup_scene`:
      - Currently **empty**, but later logic assumes `self.scene["boxes"]` exists with `root_pos_w` and `root_quat_w`.
      - This is a critical structural hole (C) in the current implementation.
    - `_get_observations`:
      - Reads `root_pos_w` and `root_quat_w` from `self.scene["boxes"]` when present.
      - Falls back to zero tensors (with identity quaternion) when `boxes` view is missing.
      - Calls `WarpHeightmapGenerator.forward` to generate a `(N, H, W)` heightmap.
      - Flattens and concatenates:
        - heightmap,
        - flattened buffer,
        - current box dims,
        - placeholder proprio (zeros),
        - and returns `{"policy": obs, "critic": obs}` with shape `(N, 38477)`.
      - **Quaternion convention**: directly forwards `root_quat_w` into Warp without explicit re‑ordering, despite Warp expecting `(x, y, z, w)` and Isaac typically using `(w, x, y, z)` → critical issue (D).
    - `_get_rewards` / `_get_dones`:
      - Reward:
        - Penalty for store operations and increasing buffer age.
        - Rewards for successful retrieve and stable placements.
        - Large negative reward when a placed box falls or is unstable.
        - Volume‑proportional bonus for successful placements.
      - Done:
        - Episode termination when a placed box falls/unstable (via current box pose vs. `last_target_pos`).
        - Termination when `box_idx >= max_boxes`.
        - Truncation: based on `episode_length_buf >= max_episode_length`.
    - `_reset_idx`:
      - Partial reset of per‑env tensors and randomization of `box_dims`.
    - `_apply_action`:
      - Interprets `action` as `[op_type, slot_idx, grid_x, grid_y, rot_idx]`.
      - Computes target world pose for the current box and writes it into the sim via `self.scene["boxes"].write_root_pose_to_sim`.
      - Uses fixed grid spacing and a high drop height.
      - Builds quaternions assuming an Isaac‑style `(w, x, y, z)` order.
      - Updates buffer store/retrieve state through `_handle_buffer_actions`.
    - `step`:
      - Custom implementation, bypassing DirectRLEnv’s usual `_pre_physics_step` / `_post_physics_step` flow:
        - Calls `_apply_action`.
        - Steps physics in a hard‑coded 50‑iteration loop.
        - Increments `box_idx`.
        - Computes obs/reward/dones via helper methods.
        - Performs in‑place resets (`_reset_idx`) for `terminated | truncated` envs.
      - Structurally works like a standard Gym‑style step but may diverge from Isaac Lab’s recommended DirectRLEnv patterns.

- **Legacy / unused env utilities**:
  - `pallet_rl.envs.heightmap_channels.compute_channels`:
    - CPU‑only numpy/scipy implementation for 2D proxy channels (support, roughness, density, etc.) from a heightmap.
    - Not referenced by the current DirectRLEnv + Warp pipeline.

### Models / Policies

- **Canonical policy**: `pallet_rl.models.rsl_rl_wrapper.PalletizerActorCritic`
  - Inherits from `rsl_rl.modules.ActorCritic` but manually re‑implements its internals using `torch.nn.Module` instead of calling `super().__init__`.
  - **Observation handling**:
    - Assumes:
      - `image_dim = 160 * 240 = 38400`
      - `vector_dim = 53` (buffer 50 + box dims 3)
      - Proprio (24 dims) is ignored.
    - Splits flattened obs into:
      - Image: reshaped to `(N, 1, 160, 240)` → CNN encoder.
      - Vector: `(N, 53)` → MLP encoder.
  - **Network**:
    - CNN: three conv layers with ELU, downsampling to `(64, 20, 30)`, then linear to a 256‑dim latent.
    - MLP: 53 → 128 → 64 with ELU.
    - Fusion: concat to 320‑dim.
    - Actor head: 320 → 128 → `55` logits (sum of MultiDiscrete dims: `3 + 10 + 16 + 24 + 2`).
    - Critic head: 320 → 128 → 1 value.
  - **MultiDiscrete distribution handling**:
    - `self.action_dims = [3, 10, 16, 24, 2]`.
    - `act`:
      - Splits logits by dimension.
      - Creates one `Categorical` per dimension.
      - Samples each independently and stacks into `(N, 5)` actions.
    - `evaluate`:
      - Recomputes logits and stores distributions for later log‑prob/entropy computation.
    - `get_actions_log_prob`:
      - Sums log‑probs from each per‑dim `Categorical`.
    - `entropy`:
      - Sums per‑dim entropies.
    - Deterministic path: `act_inference` takes per‑dim argmax.
  - **Known limitations**:
    - No explicit support for action masks yet (though structure would allow it).
    - Ignores proprioceptive features in the observation.

### Utilities

- `pallet_rl.utils.heightmap_rasterizer.WarpHeightmapGenerator`
  - Wraps a Warp kernel (`rasterize_heightmap_kernel`) to compute heightmaps on GPU.
  - Expects:
    - `box_positions` `(N * max_boxes, 3)`
    - `box_orientations` `(N * max_boxes, 4)` as Warp quaternions.
    - `box_dimensions` `(N * max_boxes, 3)`
    - `pallet_positions` `(N, 3)`
    - `pallet_orientations` `(N, 4)`
  - The kernel uses `wp.quat_rotate` / `wp.quat_rotate_inv`, which internally assume Warp’s quaternion order `(x, y, z, w)`.
  - Current env code passes Isaac quaternions directly with no conversion → convention mismatch (D).

- `pallet_rl.algo.utils`
  - `load_config(config_path)`: thin YAML loader.
  - `decode_action(action_idx, width, height, num_rotations)`:
    - Rectangular‑grid‑safe decoding of a flat index into `(rot, x, y)`.
    - Used only by tests at the moment; not wired into the new MultiDiscrete env.

### Configs

- `pallet_rl/configs/base.yaml`
  - Legacy configuration for an older PPO pipeline (non‑RSL‑RL).
  - Defines:
    - `env.grid`, `model.encoder2d`, `model.unet2d`, etc.
  - References models and env APIs that no longer exist.

- `pallet_rl/configs/rsl_rl_config.yaml`
  - Modern RSL‑RL configuration:
    - Top‑level `runner`, `policy`, `algorithm`, `env` sections.
    - Aligned with the current DirectRLEnv obs/action dimensions.
  - **Not yet loaded** by `scripts/train.py`, which instead builds a separate in‑code config.

### Tests

- `tests/test_imports.py`
  - Confirms basic `pallet_rl` import.
  - Tries to import several legacy submodules (`models.actor_critic`, `models.policy_heads`) which are **missing** in this repo revision.

- `tests/test_bugs.py`
  - Contains static tests for:
    - `algo.utils.decode_action` on rectangular grids.
    - Legacy `ActorCritic` and `SpatialPolicyHead` implementations (which are no longer present).
    - `PalletizerActorCritic` entropy / log‑prob behavior.
  - Result: only a subset of these tests are actually meaningful for the current code; others reference non‑existent modules.

### Known Structural / Logical Issues (As Is)

1. **Broken / stale imports in tests**:
   - `tests/test_imports.py` and parts of `tests/test_bugs.py` import modules that no longer exist.
2. **Incomplete scene setup**:
   - `_setup_scene` is effectively a stub; downstream code assumes a `boxes` view exists in `self.scene`.
3. **Quaternion convention mismatch**:
   - Isaac Lab’s `root_quat_w` (likely `(w, x, y, z)`) is passed directly into Warp, which expects `(x, y, z, w)`.
4. **DirectRLEnv usage divergence**:
   - Custom `step` implementation bypasses some of DirectRLEnv’s typical extension points.
5. **Config hygiene issues**:
   - Two parallel PPO configs: the in‑code dict in `scripts/train.py` and `configs/rsl_rl_config.yaml`.
   - `base.yaml` and mask/U‑Net configs are legacy but still present, creating confusion.
6. **Legacy evaluation / dataset scripts**:
   - `scripts/eval.py` and `scripts/gen_mask_dataset.py` depend on missing modules and an older pipeline (U‑Net + spatial head), conflicting with the current RSL‑RL setup.

These are the starting conditions that subsequent refactors will address while converging on a single, modern Isaac Lab DirectRLEnv + RSL‑RL PPO pipeline with Warp‑based heightmaps and a clean MultiDiscrete policy.

