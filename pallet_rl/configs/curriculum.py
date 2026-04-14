"""
Curriculum stages for the palletizing environment.

Defines progressive difficulty stages that control:
  - action space (joint spatial vs. factored)
  - spatial grid resolution
  - box count and stacking difficulty
  - reward shaping weights
  - placement constraints
  - PPO training hyperparameters

Usage::

    from pallet_rl.configs.curriculum import CurriculumStage, apply_stage_to_cfg
    apply_stage_to_cfg(CurriculumStage.STAGE_A, env_cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # PalletTaskCfg imported lazily to avoid Isaac Lab at module-load


class CurriculumStage(Enum):
    """Training curriculum stages, ordered by difficulty."""
    STAGE_A = "A"  # Learn valid placement (place-only, coarse grid)
    STAGE_B = "B"  # Improve filling quality (place-only, medium grid)
    STAGE_C = "C"  # Near-full palletization (place-only, fine grid)
    STAGE_D = "D"  # Advanced: buffer ops re-enabled, factored actions


@dataclass
class StageConfig:
    """All stage-dependent parameters in one place."""

    # --- Action Space ---
    place_only: bool = True
    grid_x: int = 8
    grid_y: int = 12
    num_rotations: int = 2

    # --- Task ---
    num_boxes: int = 8
    max_stack_height: float = 0.6  # near single-layer for Stage A
    max_episode_steps: int = 200

    # --- Placement Constraints ---
    place_support_ratio_min: float = 0.45
    place_support_height_tol_m: float = 0.02

    # --- Perception ---
    heightmap_source: str = "warp"
    depth_noise_enable: bool = False

    # --- Rewards (positive) ---
    reward_place_progress_scale: float = 15.0
    reward_packing_density_scale: float = 0.0  # disabled by default in early stages
    reward_success_scale: float = 200.0
    reward_stable: float = 3.0

    # --- Penalties (negative) ---
    penalty_time_scale: float = -0.01
    penalty_inactivity_scale: float = -0.1
    penalty_repetition_scale: float = -0.05
    penalty_stagnation_scale: float = -2.0

    # --- Stagnation ---
    stagnation_window: int = 50

    # --- PPO / Training ---
    entropy_coef: float = 0.02
    num_steps_per_env: int = 48

    @property
    def total_place_actions(self) -> int:
        """Total number of joint (x, y, rot) placement actions."""
        return self.grid_x * self.grid_y * self.num_rotations


# =========================================================================
# Stage Defaults
# =========================================================================

STAGE_CONFIGS: dict[CurriculumStage, StageConfig] = {
    CurriculumStage.STAGE_A: StageConfig(
        place_only=True,
        grid_x=8, grid_y=12, num_rotations=2,
        num_boxes=8,
        max_stack_height=0.6,
        max_episode_steps=200,
        place_support_ratio_min=0.45,
        heightmap_source="warp",
        depth_noise_enable=False,
        reward_place_progress_scale=15.0,
        reward_packing_density_scale=0.0,
        reward_success_scale=200.0,
        reward_stable=3.0,
        penalty_time_scale=-0.01,
        penalty_inactivity_scale=-0.1,
        penalty_repetition_scale=-0.05,
        penalty_stagnation_scale=-2.0,
        stagnation_window=50,
        entropy_coef=0.02,
        num_steps_per_env=48,
    ),

    CurriculumStage.STAGE_B: StageConfig(
        place_only=True,
        grid_x=12, grid_y=18, num_rotations=2,
        num_boxes=16,
        max_stack_height=1.0,
        max_episode_steps=300,
        place_support_ratio_min=0.50,
        heightmap_source="warp",
        depth_noise_enable=False,
        reward_place_progress_scale=12.0,
        reward_packing_density_scale=1.0,
        reward_success_scale=150.0,
        reward_stable=2.0,
        penalty_time_scale=-0.02,
        penalty_inactivity_scale=-0.15,
        penalty_repetition_scale=-0.1,
        penalty_stagnation_scale=-3.0,
        stagnation_window=50,
        entropy_coef=0.015,
        num_steps_per_env=48,
    ),

    CurriculumStage.STAGE_C: StageConfig(
        place_only=True,
        grid_x=16, grid_y=24, num_rotations=2,
        num_boxes=30,
        max_stack_height=1.5,
        max_episode_steps=400,
        place_support_ratio_min=0.55,
        heightmap_source="warp",
        depth_noise_enable=False,
        reward_place_progress_scale=10.0,
        reward_packing_density_scale=2.0,
        reward_success_scale=100.0,
        reward_stable=1.5,
        penalty_time_scale=-0.03,
        penalty_inactivity_scale=-0.2,
        penalty_repetition_scale=-0.15,
        penalty_stagnation_scale=-4.0,
        stagnation_window=50,
        entropy_coef=0.01,
        num_steps_per_env=36,
    ),

    CurriculumStage.STAGE_D: StageConfig(
        place_only=False,
        # Stage D uses the original factored action space (3,10,16,24,2).
        # grid_x/grid_y/num_rotations are only used for place-only stages.
        grid_x=16, grid_y=24, num_rotations=2,
        num_boxes=50,
        max_stack_height=1.8,
        max_episode_steps=200,
        place_support_ratio_min=0.60,
        heightmap_source="warp",
        depth_noise_enable=True,
        reward_place_progress_scale=5.0,
        reward_packing_density_scale=2.0,
        reward_success_scale=50.0,
        reward_stable=1.0,
        penalty_time_scale=-0.05,
        penalty_inactivity_scale=-0.3,
        penalty_repetition_scale=-0.2,
        penalty_stagnation_scale=-5.0,
        stagnation_window=50,
        entropy_coef=0.01,
        num_steps_per_env=24,
    ),
}


def get_stage(name: str) -> CurriculumStage:
    """Resolve a stage name string ('A', 'B', 'C', 'D') to enum."""
    name = name.strip().upper()
    for stage in CurriculumStage:
        if stage.value == name:
            return stage
    valid = [s.value for s in CurriculumStage]
    raise ValueError(f"Unknown curriculum stage '{name}'. Valid: {valid}")


def get_stage_config(stage: CurriculumStage) -> StageConfig:
    """Return the StageConfig for a given stage."""
    return STAGE_CONFIGS[stage]


def apply_stage_to_cfg(stage: CurriculumStage, cfg) -> None:
    """
    Mutate a PalletTaskCfg in-place to match the given curriculum stage.

    This is the canonical way to configure an environment for a specific
    training phase.  All stage-dependent fields are overwritten; fields
    not governed by the curriculum (e.g., sim physics, visual settings)
    are left untouched.
    """
    sc = STAGE_CONFIGS[stage]

    # Action space
    cfg.place_only = sc.place_only
    cfg.place_only_grid = (sc.grid_x, sc.grid_y)
    cfg.place_only_rotations = sc.num_rotations

    # Task
    cfg.num_boxes = sc.num_boxes
    cfg.max_stack_height = sc.max_stack_height
    cfg.max_episode_steps = sc.max_episode_steps

    # Placement constraints
    cfg.place_support_ratio_min = sc.place_support_ratio_min
    cfg.place_support_height_tol_m = sc.place_support_height_tol_m

    # Perception
    cfg.heightmap_source = sc.heightmap_source
    cfg.depth_noise_enable = sc.depth_noise_enable

    # Rewards
    cfg.reward_place_progress_scale = sc.reward_place_progress_scale
    cfg.reward_packing_density_scale = sc.reward_packing_density_scale
    cfg.reward_success_scale = sc.reward_success_scale
    cfg.reward_stable = sc.reward_stable

    # Penalties
    cfg.penalty_time_scale = sc.penalty_time_scale
    cfg.penalty_inactivity_scale = sc.penalty_inactivity_scale
    cfg.penalty_repetition_scale = sc.penalty_repetition_scale
    cfg.penalty_stagnation_scale = sc.penalty_stagnation_scale

    # Stagnation
    cfg.stagnation_window = sc.stagnation_window
