# PalletRL Architecture

## Overview

GPU-only palletizing RL environment built on [Isaac Lab 4.0+](https://github.com/NVIDIA-Omniverse/IsaacLab) with [RSL-RL](https://github.com/leggedrobotics/rsl_rl) PPO training.

## Package Layout

```
pallet_rl/
├── pallet_rl/
│   ├── __init__.py          # Lazy imports (no Isaac/Warp at import time)
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── pallet_task.py           # Thin orchestrator (configs + lifecycle)
│   │   ├── scene_builder.py         # Scene setup, lighting, pallet mesh, mockup physics
│   │   ├── observation_builder.py   # Heightmap + obs vector construction
│   │   ├── reward_manager.py        # All reward terms, KPI logging, settling
│   │   ├── buffer_logic.py          # Store/retrieve buffer operations
│   │   ├── placement_controller.py  # Action decode, height validation, pose writing
│   │   └── perception/
│   │       ├── __init__.py          # Factory: create_backend("warp"|"depth_camera")
│   │       ├── base.py              # BaseHeightmapBackend ABC
│   │       ├── warp_backend.py      # Warp analytical rasterizer adapter
│   │       └── depth_camera_backend.py  # Depth camera → heightmap adapter
│   ├── models/
│   │   └── rsl_rl_wrapper.py        # CNN-based ActorCritic for RSL-RL
│   ├── utils/
│   │   ├── heightmap_rasterizer.py  # WarpHeightmapGenerator (GPU Warp kernel)
│   │   ├── depth_heightmap.py       # DepthHeightmapConverter (GPU scatter-max)
│   │   ├── quaternions.py           # wxyz↔xyzw conversions
│   │   ├── usd_helpers.py           # Shared USD/PhysX helpers
│   │   └── device_utils.py          # CUDA device selection
│   ├── algo/
│   │   └── utils.py                 # Legacy utilities (config, decode)
│   └── configs/
│       └── rsl_rl_config.yaml       # RSL-RL PPO hyperparameters
├── scripts/
│   ├── train.py                     # Canonical training pipeline
│   ├── eval.py                      # Canonical evaluation pipeline
│   └── mockup_video_physics.py      # Physics-based mockup video generator
├── tests/
│   ├── test_modular.py              # Tests for extracted modules
│   ├── test_quaternions.py          # Quaternion conversion tests
│   ├── test_constraints.py          # Mass/height constraint tests
│   ├── test_imports.py              # Import hygiene tests
│   └── ...
└── legacy/                          # Archived deprecated code
```

## Data Flow

```
Action (5-dim) ──┐
                 │
    ┌────────────▼────────────┐
    │  placement_controller   │──── height validation
    │  (decode + pose write)  │──── buffer_logic (store/retrieve)
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │     PhysX Simulation    │──── settling, contact forces
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │  observation_builder    │──── perception backend
    │  (heightmap + concat)   │     (Warp raster or depth cam)
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │    reward_manager       │──── KPI logging to TensorBoard
    │  (rewards + settling)   │
    └─────────────────────────┘
```

## Action Space

The environment relies on a strict separation between the trainer-facing API and the semantic action interpretation:

1. **Trainer-Facing Interface**: A continuous `gym.spaces.Box(low=-1.0, high=1.0, shape=(5,))` exposed for RSL-RL compatibility.
2. **Decoding Layer**: `pallet_rl.envs.action_adapter` decodes the continuous `[-1, 1]` float tensor into a `DecodedAction` dataclass.
3. **Semantic Action Space**: Internal logic consumes the factored discrete 5-tuple:

| Index | Name      | Values     | Semantic                |
|-------|-----------|------------|-------------------------|
| 0     | Operation | 0/1/2      | Place / Store / Retrieve |
| 1     | Slot      | 0–9        | Buffer slot index        |
| 2     | Grid X    | 0–15       | Pallet X position        |
| 3     | Grid Y    | 0–23       | Pallet Y position        |
| 4     | Rotation  | 0/1        | 0° or 90° Z-rotation    |

Grid → World: pallet (1.2×0.8 m) centered at origin.

## Perception Backends

The environment supports swappable perception backends via `envs/perception/`:

1. `WarpBackend`: Uses `heightmap_rasterizer` (analytical, exact, fast GPU raycasts).
2. `DepthCameraBackend`: Uses Isaac Lab camera sensors + coordinate projection (realistic, noisy, handles occlusions).

Both conform to `BaseHeightmapBackend`, keeping `observation_builder` clean.

## RSL-RL Integration Workaround

**Note to Contributors:** The integration of the CNN-based `PalletizerActorCritic` into RSL-RL is handled via a **monkey-patch workaround**, not a native dependency injection framework.

RSL-RL (v1/v2) hardcodes policy lookups against its own internal `rsl_rl.modules` namespace. To inject our custom image+vector fusion network without forking the trainer, we provide a centralized hook:

```python
# pallet_rl.models.rsl_rl_wrapper
def register_custom_policy():
    import rsl_rl.modules
    rsl_rl.modules.ActorCritic = PalletizerActorCritic
```

All entrypoint scripts (`train.py`, `eval.py`) must call this hook before instantiating the `OnPolicyRunner`.
This is a conscious, contained compatibility measure. It is not an ideal long-term architecture, but it cleanly isolates the hack in a single maintainable file rather than polluting the scripts inline.

## Key Design Decisions

1. **Lazy imports**: `envs/__init__.py` and `utils/__init__.py` use try/except to avoid loading Isaac Lab or Warp at import time, enabling lightweight testing.

2. **Thin orchestrator**: `pallet_task.py` (~530 lines) delegates all heavy logic to focused modules via one-line methods.

3. **Perception abstraction**: `BaseHeightmapBackend` ABC enables swapping heightmap sources without modifying observation construction.

4. **RSL-RL policy registration**: Centralized compatibility workaround — documented and contained, rather than a native dependency injection framework.

5. **GPU-only data flow**: All tensors, heightmaps, and observations stay on CUDA. No CPU copies in the training loop.
