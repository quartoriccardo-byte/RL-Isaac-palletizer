"""
Unit tests for the curriculum stage system.
Does NOT require Isaac Lab.
"""

import pytest
from pallet_rl.configs.curriculum import (
    CurriculumStage,
    StageConfig,
    STAGE_CONFIGS,
    get_stage,
    get_stage_config,
    apply_stage_to_cfg,
)


class MockTaskCfg:
    """Minimal stand-in for PalletTaskCfg without Isaac Lab imports."""
    def __init__(self):
        self.place_only = False
        self.place_only_grid = (16, 24)
        self.place_only_rotations = 2
        self.num_boxes = 50
        self.max_stack_height = 1.8
        self.max_episode_steps = 200
        self.place_support_ratio_min = 0.60
        self.place_support_height_tol_m = 0.02
        self.heightmap_source = "warp"
        self.depth_noise_enable = True
        self.reward_place_progress_scale = 5.0
        self.reward_packing_density_scale = 2.0
        self.reward_success_scale = 50.0
        self.reward_stable = 1.0
        self.penalty_time_scale = -0.05
        self.penalty_inactivity_scale = -0.3
        self.penalty_repetition_scale = -0.2
        self.penalty_stagnation_scale = -5.0
        self.stagnation_window = 50


# =========================================================================
# Stage enum resolution
# =========================================================================

class TestGetStage:
    def test_valid_stages(self):
        assert get_stage("A") == CurriculumStage.STAGE_A
        assert get_stage("B") == CurriculumStage.STAGE_B
        assert get_stage("C") == CurriculumStage.STAGE_C
        assert get_stage("D") == CurriculumStage.STAGE_D

    def test_case_insensitive(self):
        assert get_stage("a") == CurriculumStage.STAGE_A
        assert get_stage("d") == CurriculumStage.STAGE_D

    def test_invalid_stage(self):
        with pytest.raises(ValueError, match="Unknown curriculum stage"):
            get_stage("E")

    def test_whitespace(self):
        assert get_stage("  A ") == CurriculumStage.STAGE_A


# =========================================================================
# Stage config defaults
# =========================================================================

class TestStageConfigs:
    def test_all_stages_defined(self):
        for stage in CurriculumStage:
            assert stage in STAGE_CONFIGS

    def test_stage_a_is_place_only(self):
        cfg = STAGE_CONFIGS[CurriculumStage.STAGE_A]
        assert cfg.place_only is True

    def test_stage_d_is_not_place_only(self):
        cfg = STAGE_CONFIGS[CurriculumStage.STAGE_D]
        assert cfg.place_only is False

    def test_stage_a_grid(self):
        cfg = STAGE_CONFIGS[CurriculumStage.STAGE_A]
        assert cfg.grid_x == 8
        assert cfg.grid_y == 12
        assert cfg.num_rotations == 2

    def test_stage_b_grid(self):
        cfg = STAGE_CONFIGS[CurriculumStage.STAGE_B]
        assert cfg.grid_x == 12
        assert cfg.grid_y == 18

    def test_stage_c_grid(self):
        cfg = STAGE_CONFIGS[CurriculumStage.STAGE_C]
        assert cfg.grid_x == 16
        assert cfg.grid_y == 24

    def test_total_place_actions(self):
        a = STAGE_CONFIGS[CurriculumStage.STAGE_A]
        assert a.total_place_actions == 8 * 12 * 2  # 192

        b = STAGE_CONFIGS[CurriculumStage.STAGE_B]
        assert b.total_place_actions == 12 * 18 * 2  # 432

        c = STAGE_CONFIGS[CurriculumStage.STAGE_C]
        assert c.total_place_actions == 16 * 24 * 2  # 768

    def test_stage_a_rewards(self):
        cfg = STAGE_CONFIGS[CurriculumStage.STAGE_A]
        assert cfg.reward_place_progress_scale == 15.0
        assert cfg.reward_success_scale == 200.0
        assert cfg.reward_stable == 3.0
        assert cfg.reward_packing_density_scale == 0.0

    def test_stage_a_penalties(self):
        cfg = STAGE_CONFIGS[CurriculumStage.STAGE_A]
        assert cfg.penalty_time_scale == -0.01
        assert cfg.penalty_inactivity_scale == -0.1
        assert cfg.penalty_repetition_scale == -0.05
        assert cfg.penalty_stagnation_scale == -2.0

    def test_stage_a_ppo(self):
        cfg = STAGE_CONFIGS[CurriculumStage.STAGE_A]
        assert cfg.entropy_coef == 0.02
        assert cfg.num_steps_per_env == 48

    def test_difficulty_increases_across_stages(self):
        """Verify progression: more boxes, higher stack, tighter support."""
        a = STAGE_CONFIGS[CurriculumStage.STAGE_A]
        b = STAGE_CONFIGS[CurriculumStage.STAGE_B]
        c = STAGE_CONFIGS[CurriculumStage.STAGE_C]
        d = STAGE_CONFIGS[CurriculumStage.STAGE_D]

        assert a.num_boxes < b.num_boxes < c.num_boxes <= d.num_boxes
        assert a.max_stack_height < b.max_stack_height < c.max_stack_height <= d.max_stack_height
        assert a.place_support_ratio_min <= b.place_support_ratio_min <= c.place_support_ratio_min


# =========================================================================
# apply_stage_to_cfg
# =========================================================================

class TestApplyStageToCfg:
    def test_sets_place_only(self):
        cfg = MockTaskCfg()
        apply_stage_to_cfg(CurriculumStage.STAGE_A, cfg)
        assert cfg.place_only is True

    def test_preserves_non_curriculum_fields(self):
        """Fields not governed by curriculum should remain untouched."""
        cfg = MockTaskCfg()
        cfg.sim_device = "cuda:1"  # not in curriculum
        apply_stage_to_cfg(CurriculumStage.STAGE_A, cfg)
        assert cfg.sim_device == "cuda:1"

    def test_stage_d_disables_place_only(self):
        cfg = MockTaskCfg()
        apply_stage_to_cfg(CurriculumStage.STAGE_A, cfg)
        assert cfg.place_only is True
        apply_stage_to_cfg(CurriculumStage.STAGE_D, cfg)
        assert cfg.place_only is False

    def test_grid_overridden(self):
        cfg = MockTaskCfg()
        apply_stage_to_cfg(CurriculumStage.STAGE_A, cfg)
        assert cfg.place_only_grid == (8, 12)
        assert cfg.place_only_rotations == 2

    def test_reward_overridden(self):
        cfg = MockTaskCfg()
        apply_stage_to_cfg(CurriculumStage.STAGE_A, cfg)
        assert cfg.reward_place_progress_scale == 15.0
        assert cfg.reward_packing_density_scale == 0.0
        assert cfg.reward_success_scale == 200.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
