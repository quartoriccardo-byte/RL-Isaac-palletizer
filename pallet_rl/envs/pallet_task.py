"""
PalletTask: Isaac Lab 4.0+ DirectRLEnv Implementation

GPU-only palletizing environment for RL training with RSL-RL.
Uses Warp heightmap rasterizer for vision observations.

Architecture (Post-Refactor):
    PalletTask is a thin orchestrator that delegates to:
    - scene_builder       : scene setup, lighting, pallet mesh, mockup physics
    - observation_builder : heightmap generation and obs concatenation
    - reward_manager      : all reward terms, KPI tracking, settling evaluation
    - placement_controller: action decoding, height validation, pose writing
    - buffer_logic        : store/retrieve operations, mass bookkeeping
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, Any

# Isaac Lab imports (4.0+ namespace)
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, RenderCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.sim.schemas import RigidBodyPropertiesCfg, CollisionPropertiesCfg, MassPropertiesCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
try:
    from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
except ImportError:
    try:
        from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
    except ImportError:
        from isaaclab.sim.spawners.materials.physics_materials import RigidBodyMaterialCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.sensors import PinholeCameraCfg

import gymnasium as gym

# NOTE: WarpHeightmapGenerator and DepthHeightmapConverter are imported
# lazily inside __init__ to avoid pulling in Warp/CUDA at module load time.
# This prevents Warp driver API errors in RGB-only mockup runs.
from pallet_rl.utils.quaternions import wxyz_to_xyzw, quat_angle_deg

# Extracted modules
from pallet_rl.envs.scene_builder import setup_scene, _create_prim
from pallet_rl.envs.observation_builder import build_observations
from pallet_rl.envs.reward_manager import compute_rewards
from pallet_rl.envs.placement_controller import (
    pre_physics_step,
    apply_action,
    get_action_mask as _get_action_mask_impl,
)


# =============================================================================
# Scene Configuration
# =============================================================================

_DEFAULT_MAX_BOXES = 50
_DEFAULT_PALLET_SIZE = (1.2, 0.8)


@configclass
class PalletSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the palletizing environment."""

    pallet: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pallet",
        spawn=CuboidCfg(
            size=(_DEFAULT_PALLET_SIZE[0], _DEFAULT_PALLET_SIZE[1], 0.15),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(),
            mass_props=MassPropertiesCfg(mass=25.0),
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.8, restitution=0.02,
                friction_combine_mode="max", restitution_combine_mode="min",
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.075), rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    boxes: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            f"box_{i}": RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Boxes/box_{i}",
                spawn=CuboidCfg(
                    size=(0.4, 0.3, 0.2),
                    rigid_props=RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=2,
                        linear_damping=0.1, angular_damping=0.2,
                    ),
                    collision_props=CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                    mass_props=MassPropertiesCfg(density=250.0),
                    physics_material=RigidBodyMaterialCfg(
                        static_friction=1.0, dynamic_friction=0.8, restitution=0.02,
                        friction_combine_mode="max", restitution_combine_mode="min",
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 1.5), rot=(1.0, 0.0, 0.0, 0.0),
                    lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0),
                ),
            )
            for i in range(_DEFAULT_MAX_BOXES)
        },
    )

    render_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RenderCamera",
        spawn=PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(pos=(2.5, 2.5, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        width=1280, height=720, data_types=["rgb"], update_period=0.0,
    )

    depth_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DepthCamera",
        spawn=PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 3.0), rot=(0.0, 0.0, 1.0, 0.0), convention="ros"),
        width=240, height=160, data_types=["distance_to_image_plane"], update_period=0.0,
    )


# =============================================================================
# Task Configuration
# =============================================================================

@configclass
class PalletTaskCfg(DirectRLEnvCfg):
    """Configuration for the Palletizing task."""

    sim: SimulationCfg = SimulationCfg(
        dt=1/60.0, render_interval=2, device="cuda",
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=1024 * 1024,
            gpu_total_aggregate_pairs_capacity=1024 * 1024,
            gpu_heap_capacity=64 * 1024 * 1024,
            gpu_temp_buffer_capacity=16 * 1024 * 1024,
        ),
        render=RenderCfg(
            dlss_mode=0, enable_dl_denoiser=False,
            carb_settings={
                "/ngx/enabled": False, "/rtx/post/dlss/enabled": False,
                "/rtx-transient/dlssg/enabled": False, "/rtx-transient/dldenoiser/enabled": False,
                "/renderer/multiGpu/enabled": False,
                "/rtx/translucency/enabled": False, "/rtx/reflections/enabled": False,
                "/rtx/indirectDiffuse/enabled": False,
            },
        ),
    )

    tensor_device: str | None = None

    scene: PalletSceneCfg = PalletSceneCfg(
        num_envs=4096, env_spacing=3.0, replicate_physics=True,
    )

    decimation: int = 50
    episode_length_s: float = 60.0

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    observation_space = gym.spaces.Dict({})

    # --- Pallet & Heightmap ---
    pallet_size: tuple[float, float] = (1.2, 0.8)
    map_shape: tuple[int, int] = (160, 240)
    grid_res: float = 0.005
    max_height: float = 2.0

    # --- Box configuration ---
    max_boxes: int = 50
    num_boxes: int = 50

    # --- Buffer ---
    buffer_slots: int = 10
    buffer_features: int = 6

    # --- KPI ---
    kpi_settle_steps: int = 3
    robot_state_dim: int = 24

    # --- Constraints ---
    max_stack_height: float = 1.8
    max_payload_kg: float = 500.0
    base_box_mass_kg: float = 5.0
    box_mass_variance: float = 2.0

    # --- Settling / Stability ---
    settle_steps: int = 10
    drift_xy_threshold: float = 0.035
    drift_rot_threshold: float = 7.0

    # --- Rewards ---
    reward_invalid_height: float = -2.0
    reward_infeasible: float = -4.0
    reward_fall: float = -25.0
    reward_drift: float = -3.0
    reward_stable: float = 1.0

    # --- Visual Features ---
    use_pallet_mesh_visual: bool = False
    pallet_mesh_stl_path: str = "assets/EuroPalletH0_2.STL"
    pallet_mesh_scale: tuple[float, float, float] = (0.001, 0.001, 0.001)
    pallet_mesh_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pallet_mesh_offset_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    pallet_mesh_cache_dir: str = "assets/_usd_cache"
    pallet_mesh_auto_center: bool = True
    pallet_mesh_auto_align_z: bool = True

    floor_visual_enabled: bool = True
    floor_size_xy: tuple[float, float] = (20.0, 20.0)
    floor_thickness: float = 0.02
    floor_color: tuple[float, float, float] = (0.55, 0.53, 0.50)

    # --- Mockup Mode ---
    mockup_mode: bool = False
    mockup_box_static_friction: float = 1.5
    mockup_box_dynamic_friction: float = 1.2
    mockup_box_restitution: float = 0.0
    mockup_box_linear_damping: float = 2.0
    mockup_box_angular_damping: float = 2.0
    mockup_box_max_linear_velocity: float = 2.0
    mockup_box_max_angular_velocity: float = 10.0
    mockup_solver_position_iterations: int = 12
    mockup_solver_velocity_iterations: int = 4
    mockup_contact_offset: float = 0.02
    mockup_rest_offset: float = 0.001
    mockup_max_depenetration_velocity: float = 0.5
    mockup_enable_ccd: bool = False
    mockup_pallet_static_friction: float = 1.0
    mockup_pallet_dynamic_friction: float = 0.8
    mockup_drop_height_m: float = 0.4

    # --- Heightmap Source ---
    heightmap_source: str = "warp"
    depth_cam_height_m: float = 3.0
    depth_cam_fov_deg: float = 40.0
    depth_cam_resolution: tuple[int, int] = (160, 240)
    depth_cam_update_period: float = 0.0
    depth_cam_decimation: int = 1
    depth_noise_enable: bool = True
    depth_noise_sigma_m: float = 0.003
    depth_noise_scale: float = 0.7
    depth_noise_quantization_m: float = 0.002
    depth_noise_dropout_prob: float = 0.001
    depth_crop_x: tuple[float, float] = (-0.65, 0.65)
    depth_crop_y: tuple[float, float] = (-0.45, 0.45)
    depth_debug_save_frames: bool = False
    depth_debug_save_dir: str = "debug/depth_frames"

    @property
    def num_observations(self) -> int:
        vis_dim = self.map_shape[0] * self.map_shape[1]
        buf_dim = self.buffer_slots * self.buffer_features
        box_dim = 3
        mass_dim = 2
        constraint_dim = 2
        return vis_dim + buf_dim + box_dim + mass_dim + constraint_dim + self.robot_state_dim

    action_dims: tuple[int, ...] = (3, 10, 16, 24, 2)

    @property
    def num_actions(self) -> int:
        return len(self.action_dims)


# =============================================================================
# Environment
# =============================================================================

class PalletTask(DirectRLEnv):
    """
    GPU-only palletizing environment for Isaac Lab 4.0+.

    Observations:
        - Heightmap: (H*W,) normalized [0, 1]
        - Buffer state: (buffer_slots * buffer_features,)
        - Current box dims: (3,)
        - Payload / mass norms: (2,)
        - Constraint norms: (2,)
        - Proprioception: (robot_state_dim,)

    Actions (Continuous Box -> Factored Discrete):
        - Operation: [0=Place, 1=Store, 2=Retrieve]
        - Buffer slot: [0..9]
        - Grid X: [0..15]
        - Grid Y: [0..23]
        - Rotation: [0=0°, 1=90°]
    """

    cfg: PalletTaskCfg

    def __init__(self, cfg: PalletTaskCfg, render_mode: str | None = None, **kwargs):
        self._device = cfg.sim.device

        # Create container prims BEFORE scene construction
        env_ns = getattr(cfg.scene, "env_ns", "/World/envs")
        _create_prim(f"{env_ns}/env_0/Boxes", "Xform")

        # Set action space before super().__init__()
        cfg.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        super().__init__(cfg, render_mode, **kwargs)

        # Perception Backend Abstraction
        from pallet_rl.envs.perception import create_backend
        self._heightmap_backend = create_backend(self.cfg.heightmap_source)

        # Legacy generator — only needed for training (Warp backend).
        # In mockup_mode, heightmaps are never generated via the env pipeline,
        # so we skip the WarpHeightmapGenerator entirely to avoid loading Warp.
        if not cfg.mockup_mode:
            from pallet_rl.utils.heightmap_rasterizer import WarpHeightmapGenerator
            self.heightmap_gen = WarpHeightmapGenerator(
                device=self._device, num_envs=self.num_envs, max_boxes=self.cfg.max_boxes,
                grid_res=self.cfg.grid_res, map_shape=self.cfg.map_shape,
                pallet_dims=self.cfg.pallet_size,
            )
        else:
            self.heightmap_gen = None  # no Warp in mockup mode
            print("[INFO] Mockup mode: WarpHeightmapGenerator skipped (no Warp init)")

        self._depth_converter = None
        if self.cfg.heightmap_source == "depth_camera":
            from pallet_rl.utils.depth_heightmap import DepthHeightmapConverter, DepthHeightmapCfg
            depth_cfg = DepthHeightmapCfg(
                cam_height=self.cfg.depth_cam_resolution[0],
                cam_width=self.cfg.depth_cam_resolution[1],
                fov_deg=self.cfg.depth_cam_fov_deg,
                sensor_height_m=self.cfg.depth_cam_height_m,
                map_h=self.cfg.map_shape[0], map_w=self.cfg.map_shape[1],
                crop_x=self.cfg.depth_crop_x, crop_y=self.cfg.depth_crop_y,
                noise_enable=self.cfg.depth_noise_enable,
                noise_sigma_m=self.cfg.depth_noise_sigma_m,
                noise_scale=self.cfg.depth_noise_scale,
                noise_quantization_m=self.cfg.depth_noise_quantization_m,
                noise_dropout_prob=self.cfg.depth_noise_dropout_prob,
            )
            self._depth_converter = DepthHeightmapConverter(depth_cfg, device=self._device)
            print(f"[INFO] Depth camera heightmap pipeline enabled (res={self.cfg.depth_cam_resolution})")

        self._depth_step_count = 0
        self._cached_depth_heightmap: torch.Tensor | None = None

        # State tensors
        self._init_state_tensors()

        self.action_space = cfg.action_space

        obs_dim = getattr(self.cfg, "num_observations", None)
        if obs_dim is None:
            obs_dim = int(self._get_observations()["policy"].shape[-1])
        self.observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(int(obs_dim),), dtype=np.float32,
        )

        self._render_mode = render_mode
        if self._render_mode == "rgb_array":
            self._setup_camera_lookat()

    # =====================================================================
    # State Tensor Initialization
    # =====================================================================

    def _init_state_tensors(self):
        """Initialize all state tensors on GPU."""
        device = self._device
        n = self.num_envs

        self.box_dims = torch.zeros(n, self.cfg.max_boxes, 3, device=device)
        self._box_dims_for_hmap = torch.zeros(n, self.cfg.max_boxes, 3, device=device)
        self.buffer_state = torch.zeros(n, self.cfg.buffer_slots, self.cfg.buffer_features, device=device)
        self.buffer_has_box = torch.zeros(n, self.cfg.buffer_slots, dtype=torch.bool, device=device)
        self.buffer_box_id = torch.full((n, self.cfg.buffer_slots), -1, dtype=torch.long, device=device)
        self.box_idx = torch.zeros(n, dtype=torch.long, device=device)

        self.last_moved_box_id = torch.full((n,), -1, dtype=torch.long, device=device)
        self.active_place_mask = torch.zeros(n, dtype=torch.bool, device=device)
        self.store_mask = torch.zeros(n, dtype=torch.bool, device=device)
        self.retrieve_mask = torch.zeros(n, dtype=torch.bool, device=device)
        self.valid_retrieve = torch.zeros(n, dtype=torch.bool, device=device)
        self.valid_store = torch.zeros(n, dtype=torch.bool, device=device)

        self.last_target_pos = torch.zeros(n, 3, device=device)
        self.last_target_quat = torch.zeros(n, 4, device=device)
        self.last_target_quat[:, 0] = 1.0

        self._actions = torch.zeros(n, 5, dtype=torch.float32, device=device)
        self._inactive_box_pos = torch.tensor([1e6, 1e6, -1e6], device=device)

        # KPI settling
        self._kpi_countdown = torch.zeros(n, dtype=torch.long, device=device)
        self._kpi_pending_type = torch.zeros(n, dtype=torch.long, device=device)
        self._kpi_pending_box_id = torch.full((n,), -1, dtype=torch.long, device=device)
        self._kpi_pending_target = torch.zeros(n, 3, device=device)
        self._kpi_pending_target_quat = torch.zeros(n, 4, device=device)
        self._kpi_pending_target_quat[:, 0] = 1.0
        self._kpi_place_success_count = torch.zeros(1, device=device)
        self._kpi_place_fail_count = torch.zeros(1, device=device)
        self._kpi_retrieve_success_count = torch.zeros(1, device=device)
        self._kpi_retrieve_fail_count = torch.zeros(1, device=device)
        self._kpi_eval_count = torch.zeros(1, device=device)

        # Mass / Payload
        self.box_mass_kg = torch.zeros(n, self.cfg.max_boxes, device=device)
        self.payload_kg = torch.zeros(n, device=device)

        # Settling stability
        self._settle_countdown = torch.zeros(n, dtype=torch.long, device=device)
        self._settle_box_id = torch.full((n,), -1, dtype=torch.long, device=device)
        self._settle_target_pos = torch.zeros(n, 3, device=device)
        self._settle_target_quat = torch.zeros(n, 4, device=device)

        # Height constraint
        self._last_heightmap = None
        self._height_invalid_mask = torch.zeros(n, dtype=torch.bool, device=device)

        # Infeasibility
        self._infeasible_mask = torch.zeros(n, dtype=torch.bool, device=device)

        # KPI accumulators
        self._kpi_drift_count = torch.zeros(1, device=device)
        self._kpi_collapse_count = torch.zeros(1, device=device)
        self._kpi_infeasible_count = torch.zeros(1, device=device)
        self._kpi_stable_count = torch.zeros(1, device=device)
        self._kpi_total_drift_xy = torch.zeros(1, device=device)
        self._kpi_total_drift_deg = torch.zeros(1, device=device)
        self._kpi_total_payload = torch.zeros(1, device=device)
        self._kpi_settle_eval_count = torch.zeros(1, device=device)
        self._kpi_unstable_rot_count = torch.zeros(1, device=device)

        self.extras = {}

    # =====================================================================
    # Delegate Methods
    # =====================================================================

    def _get_box_pos_quat(self, global_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get box positions and quaternions by global flat index."""
        global_idx = global_idx.to(self._device).long()
        boxes_data = self.scene["boxes"].data
        pos = boxes_data.object_pos_w.reshape(-1, 3)[global_idx]
        quat = boxes_data.object_quat_w.reshape(-1, 4)[global_idx]
        return pos, quat

    def _setup_scene(self):
        """Delegate to scene_builder module."""
        setup_scene(self.cfg, self.scene)

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Delegate to observation_builder module."""
        return build_observations(self)

    def _get_rewards(self) -> torch.Tensor:
        """Delegate to reward_manager module."""
        return compute_rewards(self)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Delegate to placement_controller module."""
        pre_physics_step(self, actions)

    def _apply_action(self) -> None:
        """Delegate to placement_controller module."""
        apply_action(self)

    def get_action_mask(self) -> torch.Tensor:
        """Delegate to placement_controller module."""
        return _get_action_mask_impl(self)

    # =====================================================================
    # Dones & Reset (kept here — lightweight, orchestrator-level)
    # =====================================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination and truncation flags."""
        n = self.num_envs
        device = self._device

        terminated = torch.zeros(n, dtype=torch.bool, device=device)
        truncated = torch.zeros(n, dtype=torch.bool, device=device)

        valid_eval = self.last_moved_box_id >= 0

        if "boxes" in self.scene.keys() and valid_eval.any():
            valid_envs = valid_eval.nonzero(as_tuple=False).flatten()
            eval_box_idx = self.last_moved_box_id[valid_envs]
            global_idx = valid_envs * self.cfg.max_boxes + eval_box_idx
            current_pos, current_quat = self._get_box_pos_quat(global_idx)

            target_pos = self.last_target_pos[valid_envs]
            target_quat = self._settle_target_quat[valid_envs]

            dist = torch.norm(current_pos[:, :2] - target_pos[:, :2], dim=-1)
            fell = current_pos[:, 2] < 0.05

            quat_dot = (current_quat * target_quat).sum(dim=-1).abs()
            drift_rot_rad = 2 * torch.acos(quat_dot.clamp(-1 + 1e-7, 1 - 1e-7))
            drift_rot_deg = drift_rot_rad * (180.0 / 3.14159265359)

            exceeded_xy = dist > self.cfg.drift_xy_threshold
            exceeded_rot = drift_rot_deg > self.cfg.drift_rot_threshold
            unstable = exceeded_xy | exceeded_rot

            failure_valid = fell | unstable
            active_place_valid = self.active_place_mask[valid_envs]
            terminated[valid_envs] = active_place_valid & failure_valid

        terminated = terminated | (self.box_idx >= self.cfg.max_boxes)

        # Payload infeasibility
        remaining_boxes = (self.cfg.max_boxes - self.box_idx).float()
        remaining_mass = remaining_boxes * self.cfg.base_box_mass_kg
        buffer_mass = (self.buffer_state[:, :, 5] * self.buffer_has_box.float()).sum(dim=1)
        prospective_total = self.payload_kg + buffer_mass + remaining_mass
        self._infeasible_mask = prospective_total > self.cfg.max_payload_kg
        terminated = terminated | self._infeasible_mask

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Partial reset for specified environments."""
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        device = self._device

        self.buffer_state[env_ids] = 0.0
        self.buffer_has_box[env_ids] = False
        self.buffer_box_id[env_ids] = -1
        self.box_idx[env_ids] = 0
        self.last_moved_box_id[env_ids] = -1
        self.active_place_mask[env_ids] = False
        self.store_mask[env_ids] = False
        self.retrieve_mask[env_ids] = False
        self.valid_retrieve[env_ids] = False
        self.valid_store[env_ids] = False
        self.last_target_pos[env_ids] = 0.0

        self._kpi_countdown[env_ids] = 0
        self._kpi_pending_type[env_ids] = 0
        self._kpi_pending_box_id[env_ids] = -1
        self._kpi_pending_target[env_ids] = 0.0

        self.payload_kg[env_ids] = 0.0
        base_mass = self.cfg.base_box_mass_kg
        variance = self.cfg.box_mass_variance
        rand_mass = torch.rand(len(env_ids), self.cfg.max_boxes, device=device)
        self.box_mass_kg[env_ids] = base_mass + (rand_mass * 2 - 1) * variance

        self._settle_countdown[env_ids] = 0
        self._settle_box_id[env_ids] = -1
        self._settle_target_pos[env_ids] = 0.0
        self._settle_target_quat[env_ids] = 0.0
        self._settle_target_quat[env_ids, 0] = 1.0

        self._height_invalid_mask[env_ids] = False
        self._infeasible_mask[env_ids] = False

        # Randomize box dimensions
        base_dims = torch.tensor([0.4, 0.3, 0.2], device=device)
        rand_offset = torch.rand(len(env_ids), self.cfg.max_boxes, 3, device=device) * 0.2 - 0.1
        self.box_dims[env_ids] = base_dims + rand_offset

        # Move all boxes off-map
        if "boxes" in self.scene.keys():
            num_reset = len(env_ids)
            max_b = self.cfg.max_boxes
            inactive_pos = self._inactive_box_pos.unsqueeze(0).unsqueeze(0).expand(num_reset, max_b, 3)
            inactive_quat = torch.zeros(num_reset, max_b, 4, device=device)
            inactive_quat[..., 0] = 1.0
            inactive_pose = torch.cat([inactive_pos, inactive_quat], dim=-1)
            self.scene["boxes"].write_object_pose_to_sim(inactive_pose, env_ids=env_ids)

    # =====================================================================
    # Heightmap Accessors
    # =====================================================================

    def get_last_heightmap_meters(self) -> torch.Tensor | None:
        """Return last-computed heightmap in meters, shape (N, H, W)."""
        return self._last_heightmap

    def get_last_heightmap_normalized(self) -> torch.Tensor | None:
        """Return last-computed heightmap normalized to [0, 1]."""
        if self._last_heightmap is None:
            return None
        return self._last_heightmap / self.cfg.max_height

    # =====================================================================
    # Camera & Rendering
    # =====================================================================

    def _setup_camera_lookat(self):
        """Set camera pose using look-at API for robust framing."""
        if "render_camera" not in self.scene.keys():
            print("[CAMERA] No render_camera in scene, skipping look-at setup")
            return
        try:
            camera = self.scene["render_camera"]
            device = torch.device(self._device)
            eye_x, eye_y, eye_z = 2.5, 2.5, 2.5
            target_z = 0.4
            eyes = torch.tensor([[eye_x, eye_y, eye_z]], device=device).repeat(self.num_envs, 1)
            targets = torch.tensor([[0.0, 0.0, target_z]], device=device).repeat(self.num_envs, 1)
            camera.set_world_poses_from_view(eyes=eyes, targets=targets)
            print(f"[CAMERA] Set look-at pose: eye=({eye_x}, {eye_y}, {eye_z}), "
                  f"target=(0, 0, {target_z}) for {self.num_envs} envs")
        except Exception as e:
            print(f"[CAMERA WARN] Failed to set look-at pose: {e}")
            import traceback
            traceback.print_exc()

    def render(self):
        """Return RGB frame from camera sensor for video recording."""
        if self._render_mode != "rgb_array":
            return None
        if "render_camera" not in self.scene.keys():
            return None
        try:
            camera = self.scene["render_camera"]
            if hasattr(self, 'sim') and self.sim is not None:
                self.sim.render()
            camera.update(dt=self.step_dt)

            rgb_data = camera.data.output.get("rgb")
            if rgb_data is None:
                return None

            frame = rgb_data[0]
            if hasattr(frame, 'cpu'):
                frame = frame.cpu().numpy()

            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            if frame.dtype in (np.float32, np.float64):
                frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)

            return frame
        except Exception as e:
            print(f"[WARN] render() failed: {e}")
            return None

    # NOTE: step() is NOT overridden — DirectRLEnv.step() handles the lifecycle.
