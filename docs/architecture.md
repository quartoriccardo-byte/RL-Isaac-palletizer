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

MultiDiscrete 5-tuple:

| Index | Name      | Values     | Semantic                |
|-------|-----------|------------|-------------------------|
| 0     | Operation | 0/1/2      | Place / Store / Retrieve |
| 1     | Slot      | 0–9        | Buffer slot index        |
| 2     | Grid X    | 0–15       | Pallet X position        |
| 3     | Grid Y    | 0–23       | Pallet Y position        |
| 4     | Rotation  | 0/1        | 0° or 90° Z-rotation    |

Grid → World: pallet (1.2×0.8 m) centered at origin.

## Perception Backends

| Backend | Class | Usage | Noise |
|---------|-------|-------|-------|
| `warp` | `WarpBackend` → `WarpHeightmapGenerator` | Training (fastest) | None |
| `depth_camera` | `DepthCameraBackend` → `DepthHeightmapConverter` | Sim-to-Real | Gaussian + quantization + dropout |

Both produce identical `(N, H, W)` heightmap tensors via `BaseHeightmapBackend.generate(env)`.

## Key Design Decisions

1. **Lazy imports**: `envs/__init__.py` and `utils/__init__.py` use try/except to avoid loading Isaac Lab or Warp at import time, enabling lightweight testing.

2. **Thin orchestrator**: `pallet_task.py` (~530 lines) delegates all heavy logic to focused modules via one-line methods.

3. **Perception abstraction**: `BaseHeightmapBackend` ABC enables swapping heightmap sources without modifying observation construction.

4. **RSL-RL policy registration**: Standard `getattr(rsl_rl.modules, name)` pattern used by Isaac Lab itself — documented as intentional, not a hack.

5. **GPU-only data flow**: All tensors, heightmaps, and observations stay on CUDA. No CPU copies in the training loop.
