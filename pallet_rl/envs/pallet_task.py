"""
PalletTask: Isaac Lab 4.0+ DirectRLEnv Implementation

GPU-only palletizing environment for RL training with RSL-RL.
Uses Warp heightmap rasterizer for vision observations.

Architecture:
- DirectRLEnv for Isaac Lab compatibility
- MultiDiscrete action space: [Operation, BufferSlot, GridX, GridY, Rotation]
- CNN-compatible flattened heightmap observations
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
from isaaclab.sim.spawners import shapes as shape_spawners
# IsaacLab API update: use CuboidCfg for spawning box primitives
from isaaclab.sim.spawners.shapes import CuboidCfg
# Schema configs required to spawn rigid bodies with RigidBodyAPI
from isaaclab.sim.schemas import RigidBodyPropertiesCfg, CollisionPropertiesCfg, MassPropertiesCfg
# IsaacLab API update: ground plane spawner moved to from_files module
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
# Visual materials for ground plane / box coloring
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
# Physics materials (Version-safe import)
try:
    from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
    print("[INFO] Successfully imported RigidBodyMaterialCfg from physics_materials_cfg")
except ImportError:
    try:
        from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
        print("[INFO] Successfully imported RigidBodyMaterialCfg from materials")
    except ImportError:
        from isaaclab.sim.spawners.materials.physics_materials import RigidBodyMaterialCfg
        print("[INFO] Successfully imported RigidBodyMaterialCfg from physics_materials")
# Camera sensor for headless video recording
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.sensors import PinholeCameraCfg

import gymnasium as gym

from pallet_rl.utils.heightmap_rasterizer import WarpHeightmapGenerator
from pallet_rl.utils.quaternions import wxyz_to_xyzw, quat_angle_deg
from pallet_rl.utils.depth_heightmap import DepthHeightmapConverter, DepthHeightmapCfg


# =============================================================================
# Scene Configuration (Isaac Lab 5.0+ / Isaac Sim 5.0)
# =============================================================================
# IsaacLab 5.0 API update: Assets must be registered through InteractiveSceneCfg
# fields. The scene.add() method has been removed. Assets are automatically
# loaded when InteractiveScene(cfg) is created.

# Default max_boxes for scene configuration (matches PalletTaskCfg.max_boxes)
_DEFAULT_MAX_BOXES = 50
_DEFAULT_PALLET_SIZE = (1.2, 0.8)


@configclass
class PalletSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the palletizing environment.
    
    Defines all assets that will be spawned in each environment:
    - Pallet: Static rigid body at the origin
    - Boxes: Collection of rigid bodies for stacking
    """
    
    # Pallet as a kinematic rigid body at the origin of each env
    # NOTE: CuboidCfg requires rigid_props/collision_props/mass_props for RigidBodyAPI
    pallet: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pallet",
        spawn=CuboidCfg(
            size=(_DEFAULT_PALLET_SIZE[0], _DEFAULT_PALLET_SIZE[1], 0.15),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=CollisionPropertiesCfg(),
            mass_props=MassPropertiesCfg(mass=25.0),
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.02,
                friction_combine_mode="max",
                restitution_combine_mode="min",
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.075),
            rot=(1.0, 0.0, 0.0, 0.0),  # (w,x,y,z) for Isaac scene
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )
    
    # Box collection: each box has a unique name and prim_path
    # IsaacLab 5.0 API: RigidObjectCollectionCfg uses `rigid_objects` dict
    boxes: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            f"box_{i}": RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Boxes/box_{i}",
                # Size will be overridden at reset based on `box_dims`
                spawn=CuboidCfg(
                    size=(0.4, 0.3, 0.2),
                    rigid_props=RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,  # prevent contact "popping"
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=2,
                        linear_damping=0.1,
                        angular_damping=0.2,
                    ),
                    collision_props=CollisionPropertiesCfg(
                        contact_offset=0.005,
                        rest_offset=0.0,
                    ),
                    mass_props=MassPropertiesCfg(density=250.0),
                    physics_material=RigidBodyMaterialCfg(
                        static_friction=1.0,
                        dynamic_friction=0.8,
                        restitution=0.02,
                        friction_combine_mode="max",
                        restitution_combine_mode="min",
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 1.5),
                    rot=(1.0, 0.0, 0.0, 0.0),
                    lin_vel=(0.0, 0.0, 0.0),
                    ang_vel=(0.0, 0.0, 0.0),
                ),
            )
            for i in range(_DEFAULT_MAX_BOXES)
        },
    )
    
    # =====================================================================
    # Render camera: cinematic oblique view for video recording
    # =====================================================================
    # Runtime code calls set_world_poses_from_view() to aim at pallet center.
    render_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RenderCamera",
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(2.5, 2.5, 2.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
        width=1280,
        height=720,
        data_types=["rgb"],
        update_period=0.0,
    )
    
    # =====================================================================
    # Depth camera: top-down overhead view for depth-based heightmap
    # =====================================================================
    # Only used when heightmap_source="depth_camera".
    # Pose: 3m above pallet center, looking straight down.
    depth_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DepthCamera",
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 3.0),
            rot=(0.0, 0.0, 1.0, 0.0),  # 180° around Y → looking down (-Z)
            convention="ros",
        ),
        width=240,
        height=160,
        data_types=["distance_to_image_plane"],
        update_period=0.0,
    )


# =============================================================================
# Task Configuration
# =============================================================================

@configclass
class PalletTaskCfg(DirectRLEnvCfg):
    """Configuration for the Palletizing task."""
    
    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/60.0,
        render_interval=2,
        # NOTE: Use "cuda" (no index) so AppLauncher / CLI --device cuda:N
        # selects the actual GPU. Hardcoding "cuda:0" ignored the CLI flag
        # and caused PhysX to init on a non-RTX GPU (sm_61 < 7.0 required).
        device="cuda",
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=1024 * 1024,
            gpu_total_aggregate_pairs_capacity=1024 * 1024,
            gpu_heap_capacity=64 * 1024 * 1024,
            gpu_temp_buffer_capacity=16 * 1024 * 1024,
        ),
        # Render settings to disable NGX/DLSS and reduce VRAM usage
        render=RenderCfg(
            dlss_mode=0,
            enable_dl_denoiser=False,
            carb_settings={
                # Core NGX/DLSS disable
                "/ngx/enabled": False,
                "/rtx/post/dlss/enabled": False,
                "/rtx-transient/dlssg/enabled": False,
                "/rtx-transient/dldenoiser/enabled": False,
                # Disable multi-GPU to reduce VRAM pressure
                "/renderer/multiGpu/enabled": False,
                # Reduce VRAM-heavy features
                "/rtx/translucency/enabled": False,
                "/rtx/reflections/enabled": False,
                "/rtx/indirectDiffuse/enabled": False,
            },
        ),
    )
    
    # Scene configuration (IsaacLab 5.0: assets defined in PalletSceneCfg)
    scene: PalletSceneCfg = PalletSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
        replicate_physics=True,  # Required for env_0 to be source for cloning
    )
    
    # Decimation (physics steps per RL step)
    # Default kept at 50 to match the previous hardcoded behaviour in step().
    decimation: int = 50
    
    # Episode length
    episode_length_s: float = 60.0

    # NOTE: Required by IsaacLab DirectRLEnvCfg.validate() in newer versions.
    # Action space: continuous Box(5,) for rsl_rl compatibility
    # 5 dimensions: [op, slot, x, y, rot] all in [-1, 1]
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    observation_space = gym.spaces.Dict({})
    
    # =========================================================================
    # Pallet & Coordinate System Documentation
    # =========================================================================
    # Pallet dimensions in meters: (length_x, width_y)
    # The pallet is centered at origin (0, 0) in each environment.
    #
    # Coordinate frame (looking down from above):
    #   +X extends to the right (pallet length = 1.2m, from -0.6 to +0.6)
    #   +Y extends forward (pallet width = 0.8m, from -0.4 to +0.4)
    #   +Z extends upward (stacking direction)
    #
    # Action grid mapping:
    #   grid_x: 0..15 (16 cells) → world X: -0.6 to +0.6m (step = 0.075m)
    #   grid_y: 0..23 (24 cells) → world Y: -0.4 to +0.4m (step = 0.0333m)
    #   Note: step sizes differ to cover the asymmetric pallet dimensions
    #
    # Heightmap rasterization:
    #   map_shape = (H=160, W=240) pixels at grid_res = 0.005m
    #   Physical coverage: 160*0.005 = 0.8m (Y), 240*0.005 = 1.2m (X)
    #   This matches pallet_size exactly.
    # =========================================================================
    pallet_size: tuple[float, float] = (1.2, 0.8)  # meters (X, Y)
    
    # Heightmap configuration
    # H (rows) = 160 → covers 0.8m (Y axis)
    # W (cols) = 240 → covers 1.2m (X axis)
    # This matches pallet_size = (1.2, 0.8) exactly
    map_shape: tuple[int, int] = (160, 240)  # (H, W) pixels
    grid_res: float = 0.005  # 0.5cm resolution
    max_height: float = 2.0  # meters (for normalization)
    
    # Box configuration
    max_boxes: int = 50
    # Number of *active* boxes for placement (must be <= max_boxes).
    # Tensor sizes and observation dims are always based on max_boxes for
    # stability.  Boxes [num_boxes..max_boxes) are parked/deactivated.
    # Defaults to max_boxes so training code is unaffected.
    num_boxes: int = 50
    
    # Buffer configuration
    buffer_slots: int = 10
    buffer_features: int = 6  # [L, W, H, ID, Age, Mass] - increased from 5
    
    # KPI settling window (number of env steps to wait before evaluating KPIs)
    # This allows physics to settle before measuring place/retrieve success
    kpi_settle_steps: int = 3
    
    # Robot state dimension
    robot_state_dim: int = 24  # 6 pos + 6 vel + gripper etc.
    
    # =========================================================================
    # Stack Height Constraint
    # =========================================================================
    max_stack_height: float = 1.8  # meters - maximum allowed stack height
    
    # =========================================================================
    # Payload (Weight) Constraints
    # =========================================================================
    max_payload_kg: float = 500.0  # maximum total on-pallet mass in kg
    base_box_mass_kg: float = 5.0  # base mass per box in kg
    box_mass_variance: float = 2.0  # ± variance for mass randomization
    
    # =========================================================================
    # Settling / Stability Configuration
    # =========================================================================
    settle_steps: int = 10  # physics steps to wait after placement for settling (increased from 5)
    drift_xy_threshold: float = 0.035  # meters - max allowed XY drift (tightened from 0.05)
    drift_rot_threshold: float = 7.0  # degrees - max allowed rotation drift (tightened from 15)
    
    # =========================================================================
    # Reward Configuration for Constraints
    # =========================================================================
    # Reward hierarchy (most negative to positive):
    # 1. Physical collapse (box falls)     -> reward_fall = -25.0
    # 2. Infeasible episode termination    -> reward_infeasible = -4.0
    # 3. Unstable placement (drift)        -> reward_drift = -3.0
    # 4. Invalid action (height violation) -> reward_invalid_height = -2.0
    # 5. Stable successful placement       -> reward_stable = +1.0
    reward_invalid_height: float = -2.0  # penalty for attempting action exceeding height (increased from -0.5)
    reward_infeasible: float = -4.0  # moderate penalty for infeasible payload termination (reduced from -20)
    reward_fall: float = -25.0  # most severe penalty when box falls (increased from -15)
    reward_drift: float = -3.0  # penalty when box drifts but doesn't fall (increased from -2)
    reward_stable: float = 1.0  # bonus for stable placement
    
    # =========================================================================
    # Feature: Visual Pallet Mesh (STL→USD, purely visual)
    # =========================================================================
    use_pallet_mesh_visual: bool = False
    pallet_mesh_stl_path: str = "assets/EuroPalletH0_2.STL"
    pallet_mesh_scale: tuple[float, float, float] = (0.001, 0.001, 0.001)  # mm→m
    pallet_mesh_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pallet_mesh_offset_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    pallet_mesh_cache_dir: str = "assets/_usd_cache"
    pallet_mesh_auto_center: bool = True   # auto-center mesh XY on pallet collider
    pallet_mesh_auto_align_z: bool = True  # align mesh Z base to collider bottom (z=0)
    
    # =========================================================================
    # Visual Floor Configuration
    # =========================================================================
    floor_visual_enabled: bool = True
    floor_size_xy: tuple[float, float] = (20.0, 20.0)
    floor_thickness: float = 0.02
    floor_color: tuple[float, float, float] = (0.55, 0.53, 0.50)  # cement-ish
    
    # =========================================================================
    # Mockup Mode: Physics/Visual Overrides for Demo Videos
    # =========================================================================
    # When True, applies aggressive damping, friction, velocity clamping,
    # and a gentler drop height for stable, aesthetic box placements.
    # Training should keep this False (default).
    mockup_mode: bool = False
    
    # --- Mockup box physics material ---
    mockup_box_static_friction: float = 1.5
    mockup_box_dynamic_friction: float = 1.2
    mockup_box_restitution: float = 0.0
    
    # --- Mockup rigid body stability ---
    mockup_box_linear_damping: float = 2.0
    mockup_box_angular_damping: float = 2.0
    mockup_box_max_linear_velocity: float = 2.0     # m/s
    mockup_box_max_angular_velocity: float = 10.0    # rad/s
    mockup_solver_position_iterations: int = 12
    mockup_solver_velocity_iterations: int = 4
    mockup_contact_offset: float = 0.02
    mockup_rest_offset: float = 0.001
    mockup_max_depenetration_velocity: float = 0.5  # m/s, prevents popping
    mockup_enable_ccd: bool = False  # CCD disabled: incompatible with kinematic toggling
    
    # --- Mockup pallet surface friction ---
    mockup_pallet_static_friction: float = 1.0
    mockup_pallet_dynamic_friction: float = 0.8
    
    # --- Mockup drop height (lower = gentler placement) ---
    mockup_drop_height_m: float = 0.4
    
    # =========================================================================
    # Feature: Heightmap Source Selection
    # =========================================================================
    heightmap_source: str = "warp"  # "warp" (default, fast) or "depth_camera"
    
    # Depth camera config (only used when heightmap_source="depth_camera")
    depth_cam_height_m: float = 3.0
    depth_cam_fov_deg: float = 40.0  # covers ~1.4×0.93m at 3m (>1.2×0.8 pallet)
    depth_cam_resolution: tuple[int, int] = (160, 240)  # (H, W)
    depth_cam_update_period: float = 0.0  # 0 = every frame
    depth_cam_decimation: int = 1  # 1 = every step, N = skip N-1
    
    # Depth noise model (underestimated realism)
    depth_noise_enable: bool = True
    depth_noise_sigma_m: float = 0.003  # 3mm base noise
    depth_noise_scale: float = 0.7  # underestimate factor
    depth_noise_quantization_m: float = 0.002
    depth_noise_dropout_prob: float = 0.001
    
    # Heightmap crop bounds (world XY, slightly wider than pallet)
    depth_crop_x: tuple[float, float] = (-0.65, 0.65)
    depth_crop_y: tuple[float, float] = (-0.45, 0.45)
    
    # Debug: save depth frames to disk
    depth_debug_save_frames: bool = False
    depth_debug_save_dir: str = "debug/depth_frames"
    
    # Observation dimension (computed)
    @property
    def num_observations(self) -> int:
        # Heightmap (flattened) + Buffer + Box dims + Payload/Mass + Constraint norms + Proprio
        vis_dim = self.map_shape[0] * self.map_shape[1]  # 38400
        buf_dim = self.buffer_slots * self.buffer_features  # 60
        box_dim = 3
        mass_dim = 2  # payload_norm + current_box_mass_norm
        constraint_dim = 2  # max_payload_norm + max_stack_height_norm (for future domain randomization)
        return vis_dim + buf_dim + box_dim + mass_dim + constraint_dim + self.robot_state_dim  # 38491
    
    # Action space (MultiDiscrete dimensions)
    action_dims: tuple[int, ...] = (3, 10, 16, 24, 2)  # Op, Slot, X, Y, Rot
    
    @property
    def num_actions(self) -> int:
        return len(self.action_dims)


# =============================================================================
# Helper Functions (Robust prim/USD operations)
# =============================================================================

def _get_stage():
    import omni.usd
    return omni.usd.get_context().get_stage()

def _is_prim_path_valid(path: str) -> bool:
    try:
        stage = _get_stage()
        if not stage: return False
        return stage.GetPrimAtPath(path).IsValid()
    except Exception:
        return False

def _create_prim(path: str, prim_type: str, attributes: dict = None):
    try:
        stage = _get_stage()
        if not stage: return None
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            prim = stage.DefinePrim(path, prim_type)
        if attributes:
            from pxr import Sdf, Gf
            for k, v in attributes.items():
                if isinstance(v, float):
                    prim.CreateAttribute(k, Sdf.ValueTypeNames.Float).Set(v)
                elif isinstance(v, tuple) and len(v) == 3:
                    prim.CreateAttribute(k, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*v))
        return prim
    except Exception as e:
        print(f"[WARNING] Local create_prim failed for {path}: {e}")
        return None

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
        - Proprioception: (robot_state_dim,)
    
    Actions (MultiDiscrete):
        - Operation: [0=Place, 1=Store, 2=Retrieve]
        - Buffer slot: [0..9]
        - Grid X: [0..15]
        - Grid Y: [0..23]
        - Rotation: [0=0°, 1=90°]
    """
    
    cfg: PalletTaskCfg
    
    def __init__(self, cfg: PalletTaskCfg, render_mode: str | None = None, **kwargs):
        # Pre-init setup
        self._device = cfg.sim.device
        
        # IsaacLab 4.x+ fix: Create container prims BEFORE scene construction.
        # Spawners using regex-based prim_path (e.g. {ENV_REGEX_NS}/Boxes/box)
        # require the parent prim to exist in env_0 before InteractiveScene initializes.
        # NOTE: In current IsaacLab versions, env_ns is a property of InteractiveScene (not InteractiveSceneCfg).
        # InteractiveScene uses "/World/envs" as env namespace.
        env_ns = getattr(cfg.scene, "env_ns", "/World/envs")        
        _create_prim(
            f"{env_ns}/env_0/Boxes",
            "Xform"
        )
        
        # =====================================================================
        # CRITICAL: Set cfg.action_space BEFORE super().__init__()
        # =====================================================================
        # IsaacLab/RSL-RL captures action_space during DirectRLEnv.__init__().
        # If we set self.action_space after super(), RSL-RL sees the dummy
        # shape (1,) instead of the real shape (5,), causing broadcast errors.
        cfg.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        
        # Call parent
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize heightmap generator (GPU-only Warp)
        self.heightmap_gen = WarpHeightmapGenerator(
            device=self._device,
            num_envs=self.num_envs,
            max_boxes=self.cfg.max_boxes,
            grid_res=self.cfg.grid_res,
            map_shape=self.cfg.map_shape,
            pallet_dims=self.cfg.pallet_size
        )
        
        # Depth camera heightmap converter (initialized only when needed)
        self._depth_converter: DepthHeightmapConverter | None = None
        if self.cfg.heightmap_source == "depth_camera":
            depth_cfg = DepthHeightmapCfg(
                cam_height=self.cfg.depth_cam_resolution[0],
                cam_width=self.cfg.depth_cam_resolution[1],
                fov_deg=self.cfg.depth_cam_fov_deg,
                sensor_height_m=self.cfg.depth_cam_height_m,
                map_h=self.cfg.map_shape[0],
                map_w=self.cfg.map_shape[1],
                crop_x=self.cfg.depth_crop_x,
                crop_y=self.cfg.depth_crop_y,
                noise_enable=self.cfg.depth_noise_enable,
                noise_sigma_m=self.cfg.depth_noise_sigma_m,
                noise_scale=self.cfg.depth_noise_scale,
                noise_quantization_m=self.cfg.depth_noise_quantization_m,
                noise_dropout_prob=self.cfg.depth_noise_dropout_prob,
            )
            self._depth_converter = DepthHeightmapConverter(depth_cfg, device=self._device)
            print(f"[INFO] Depth camera heightmap pipeline enabled (res={self.cfg.depth_cam_resolution})")
        
        # Counter for depth decimation
        self._depth_step_count = 0
        self._cached_depth_heightmap: torch.Tensor | None = None
        
        # State tensors (all GPU-resident)
        self._init_state_tensors()
        
        # Action space: use cfg.action_space (set before super().__init__)\n        # This ensures self.action_space matches cfg.action_space exactly\n        self.action_space = cfg.action_space
        
        # Observation space
        obs_dim = getattr(self.cfg, "num_observations", None)
        if obs_dim is None:
            # fallback: infer from a dummy observation (safe after _init_state_tensors)
            obs_dim = int(self._get_observations()["policy"].shape[-1])

        self.observation_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(int(obs_dim),),
            dtype=np.float32,
        )
        
        # Store render mode for render() implementation
        self._render_mode = render_mode
        
        # Setup camera look-at pose if rendering is enabled
        if self._render_mode == "rgb_array":
            self._setup_camera_lookat()
    
    def _setup_camera_lookat(self):
        """Set camera pose using look-at API for robust framing.
        
        Uses set_world_poses_from_view() which computes the correct quaternion
        from eye position and look-at target. This avoids quaternion convention
        issues that cause gray frames (camera looking away from scene).
        
        Camera is positioned above and offset from the pallet center to show:
        - The entire pallet area
        - Boxes being placed/spawned
        """
        if "render_camera" not in self.scene.keys():
            print("[CAMERA] No render_camera in scene, skipping look-at setup")
            return
        
        try:
            camera = self.scene["render_camera"]
            device = torch.device(self._device)
            
            # Pallet center (origin per env) + some Z height bias
            # Target slightly above pallet surface to show stacking
            target_z = 0.4  # Look at 40cm above pallet surface
            
            # Eye position: oblique "3/4 top" view
            # Elevated and offset to see both pallet and falling boxes
            eye_x, eye_y, eye_z = 2.5, 2.5, 2.5
            
            # Create per-env tensors
            eyes = torch.tensor([[eye_x, eye_y, eye_z]], device=device).repeat(self.num_envs, 1)
            targets = torch.tensor([[0.0, 0.0, target_z]], device=device).repeat(self.num_envs, 1)
            
            # Apply look-at pose via IsaacLab API
            camera.set_world_poses_from_view(eyes=eyes, targets=targets)
            
            print(f"[CAMERA] Set look-at pose: eye=({eye_x}, {eye_y}, {eye_z}), "
                  f"target=(0, 0, {target_z}) for {self.num_envs} envs")
                  
        except Exception as e:
            print(f"[CAMERA WARN] Failed to set look-at pose: {e}")
            import traceback
            traceback.print_exc()

    def render(self):
        """Return RGB frame from camera sensor for video recording.
        
        In headless mode, env.render() returns black without an explicit camera.
        This method reads the camera sensor's RGB output and converts it to
        a uint8 numpy array suitable for gymnasium.wrappers.RecordVideo.
        
        Returns:
            np.ndarray: RGB frame (H, W, 3) as uint8, or None if no camera.
        """
        if self._render_mode != "rgb_array":
            return None
        
        # Check if camera sensor exists in scene
        if "render_camera" not in self.scene.keys():
            print(f"[RENDER ERR] render_camera not in scene! Available keys: {list(self.scene.keys())}")
            return None
        
        try:
            # Get render camera sensor
            camera = self.scene["render_camera"]
            
            # CRITICAL: Force a render tick BEFORE reading camera buffer
            # Without this, the camera may return stale/uninitialized data
            if hasattr(self, 'sim') and self.sim is not None:
                self.sim.render()
            
            # Force update the camera buffer
            camera.update(dt=self.step_dt)
            
            # Get RGB data - shape: (num_envs, H, W, 3) or (num_envs, H, W, 4)
            rgb_data = camera.data.output.get("rgb")
            if rgb_data is None:
                print("[RENDER DBG] rgb_data is None after camera.update()")
                return None
            
            # Only return env_0's frame for video recording
            frame = rgb_data[0]  # First environment
            
            # DEBUG: Log buffer statistics BEFORE any conversion
            if hasattr(frame, 'cpu'):
                frame_np = frame.cpu().numpy()
            else:
                frame_np = np.array(frame)
            
            # Sample for unique count (speed optimization)
            unique_count = len(np.unique(frame_np.ravel()[:10000]))
            print(f"[RENDER DBG] rgb dtype={frame_np.dtype} shape={frame_np.shape} "
                  f"min={frame_np.min():.4f} max={frame_np.max():.4f} "
                  f"mean={frame_np.mean():.4f} unique~{unique_count}")
            
            # DEBUG: Check depth buffer to verify camera sees geometry
            depth_data = camera.data.output.get("distance_to_image_plane")
            if depth_data is not None:
                depth_np = depth_data[0].cpu().numpy() if hasattr(depth_data[0], 'cpu') else np.array(depth_data[0])
                finite_mask = np.isfinite(depth_np)
                finite_count = finite_mask.sum()
                if finite_count > 0:
                    depth_min = depth_np[finite_mask].min()
                    depth_max = depth_np[finite_mask].max()
                    depth_mean = depth_np[finite_mask].mean()
                else:
                    depth_min = depth_max = depth_mean = float('inf')
                print(f"[RENDER DBG] depth finite_count={finite_count}/{depth_np.size} "
                      f"min={depth_min:.2f} max={depth_max:.2f} mean={depth_mean:.2f}")
            
            # Convert torch tensor to numpy
            if hasattr(frame, 'cpu'):
                frame = frame.cpu().numpy()
            
            # Handle RGBA -> RGB (drop alpha channel if present)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            
            # Convert float [0,1] to uint8 [0,255] if needed
            if frame.dtype in (np.float32, np.float64):
                frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
            
            return frame
            
        except Exception as e:
            print(f"[WARN] render() failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    # =========================================================================
    # Public heightmap accessors (for external scripts / recording)
    # =========================================================================

    def get_last_heightmap_meters(self) -> torch.Tensor | None:
        """Return the last-computed heightmap in real meters, shape (N, H, W).

        This is the raw (unnormalized) heightmap cached during
        ``_get_observations()``.  Values represent physical height above
        ground in meters.

        Returns ``None`` if no observation step has been executed yet.
        """
        return self._last_heightmap

    def get_last_heightmap_normalized(self) -> torch.Tensor | None:
        """Return the last-computed heightmap normalized to [0, 1], shape (N, H, W).

        Normalization uses ``cfg.max_height`` — identical to what the RL
        policy receives as observation input.

        Returns ``None`` if no observation step has been executed yet.
        """
        if self._last_heightmap is None:
            return None
        return self._last_heightmap / self.cfg.max_height

    def _init_state_tensors(self):
        """Initialize all state tensors on GPU."""
        device = self._device
        n = self.num_envs
        
        # Box dimensions tensor: (N, max_boxes, 3)
        self.box_dims = torch.zeros(n, self.cfg.max_boxes, 3, device=device)
        
        # Preallocated buffer for heightmap generation (avoids per-step clone)
        self._box_dims_for_hmap = torch.zeros(n, self.cfg.max_boxes, 3, device=device)
        
        # Buffer state: (N, buffer_slots, buffer_features) - dims, active flag, age
        self.buffer_state = torch.zeros(
            n, self.cfg.buffer_slots, self.cfg.buffer_features, device=device
        )
        
        # Physical buffer tracking: which physical box is parked in each slot
        # buffer_has_box[env, slot] = True if slot contains a parked physical box
        self.buffer_has_box = torch.zeros(n, self.cfg.buffer_slots, dtype=torch.bool, device=device)
        # buffer_box_id[env, slot] = physical box index (0..max_boxes-1) or -1 if empty
        self.buffer_box_id = torch.full((n, self.cfg.buffer_slots), -1, dtype=torch.long, device=device)
        
        # Current box index per env (tracks "next fresh box to use")
        self.box_idx = torch.zeros(n, dtype=torch.long, device=device)
        
        # Last moved/placed box ID for reward/done evaluation
        # After PLACE: set to box_idx-1 (the newly placed box)
        # After RETRIEVE: set to the retrieved physical box ID
        # After STORE: set to -1 (no placement occurred)
        self.last_moved_box_id = torch.full((n,), -1, dtype=torch.long, device=device)
        
        # Masks for reward computation
        self.active_place_mask = torch.zeros(n, dtype=torch.bool, device=device)
        self.store_mask = torch.zeros(n, dtype=torch.bool, device=device)
        self.retrieve_mask = torch.zeros(n, dtype=torch.bool, device=device)
        self.valid_retrieve = torch.zeros(n, dtype=torch.bool, device=device)
        self.valid_store = torch.zeros(n, dtype=torch.bool, device=device)  # True stores (box_idx>0 & empty slot)
        
        # Target position for stability check
        self.last_target_pos = torch.zeros(n, 3, device=device)
        # Target quaternion for rotation stability check (w,x,y,z format)
        self.last_target_quat = torch.zeros(n, 4, device=device)
        self.last_target_quat[:, 0] = 1.0  # Initialize to identity
        
        # Action buffer for Isaac Lab API compatibility
        # _pre_physics_step stores actions here, _apply_action reads them
        self._actions = torch.zeros(n, 5, dtype=torch.float32, device=device)
        
        # Off-map position for inactive boxes (far away to ensure they never pollute heightmap)
        self._inactive_box_pos = torch.tensor([1e6, 1e6, -1e6], device=device)
        
        # =====================================================================
        # KPI Settling Window State
        # =====================================================================
        # Pending countdown before evaluating KPIs (allows physics settling)
        self._kpi_countdown = torch.zeros(n, dtype=torch.long, device=device)
        # Type of pending action: 0=none, 1=place, 2=retrieve
        self._kpi_pending_type = torch.zeros(n, dtype=torch.long, device=device)
        # Physical box ID to evaluate when countdown reaches 0
        self._kpi_pending_box_id = torch.full((n,), -1, dtype=torch.long, device=device)
        # Target position saved at action time for KPI evaluation
        self._kpi_pending_target = torch.zeros(n, 3, device=device)
        # Target quaternion saved at action time for KPI rotation evaluation
        self._kpi_pending_target_quat = torch.zeros(n, 4, device=device)
        self._kpi_pending_target_quat[:, 0] = 1.0  # Initialize to identity
        
        # Running KPI accumulators (for settled evaluations)
        self._kpi_place_success_count = torch.zeros(1, device=device)
        self._kpi_place_fail_count = torch.zeros(1, device=device)
        self._kpi_retrieve_success_count = torch.zeros(1, device=device)
        self._kpi_retrieve_fail_count = torch.zeros(1, device=device)
        self._kpi_eval_count = torch.zeros(1, device=device)  # Total KPI evals
        
        # =====================================================================
        # Mass / Payload Tracking
        # =====================================================================
        # Per-box mass in kg: (N, max_boxes)
        self.box_mass_kg = torch.zeros(n, self.cfg.max_boxes, device=device)
        # Current on-pallet payload in kg: (N,)
        self.payload_kg = torch.zeros(n, device=device)
        
        # =====================================================================
        # Settling Stability State
        # =====================================================================
        # Countdown for settling window after PLACE/RETRIEVE
        self._settle_countdown = torch.zeros(n, dtype=torch.long, device=device)
        # Box ID being evaluated for settling
        self._settle_box_id = torch.full((n,), -1, dtype=torch.long, device=device)
        # Target position saved at placement for drift evaluation
        self._settle_target_pos = torch.zeros(n, 3, device=device)
        # Target quaternion saved at placement for rotation drift evaluation
        self._settle_target_quat = torch.zeros(n, 4, device=device)
        
        # =====================================================================
        # Height Constraint State
        # =====================================================================
        # Cache last heightmap for action masking (updated in _get_observations)
        self._last_heightmap = None  # Will be (N, H, W) tensor
        # Track actions that were height-invalid this step (for reward penalty)
        self._height_invalid_mask = torch.zeros(n, dtype=torch.bool, device=device)
        
        # =====================================================================
        # Infeasibility State
        # =====================================================================
        # Track envs that became infeasible this step
        self._infeasible_mask = torch.zeros(n, dtype=torch.bool, device=device)
        
        # =====================================================================
        # New KPI Accumulators for Constraints
        # =====================================================================
        self._kpi_drift_count = torch.zeros(1, device=device)
        self._kpi_collapse_count = torch.zeros(1, device=device)
        self._kpi_infeasible_count = torch.zeros(1, device=device)
        self._kpi_stable_count = torch.zeros(1, device=device)
        self._kpi_total_drift_xy = torch.zeros(1, device=device)
        self._kpi_total_drift_deg = torch.zeros(1, device=device)
        self._kpi_total_payload = torch.zeros(1, device=device)
        self._kpi_settle_eval_count = torch.zeros(1, device=device)
        self._kpi_unstable_rot_count = torch.zeros(1, device=device)  # Rotation-only unstable
        
        # Initialize extras dict for logging
        self.extras = {}
    
    def _get_box_pos_quat(self, global_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get box positions and quaternions by global flat index.
        
        Uses reshape-based indexing (not view) to avoid non-contiguous tensor errors.
        Centralizes RigidObjectCollection data access pattern.
        
        Args:
            global_idx: Flat indices (env_id * max_boxes + box_id), shape (M,), dtype=long
            
        Returns:
            pos: Position tensor (M, 3)
            quat: Quaternion tensor (M, 4) in (w,x,y,z) format
        """
        global_idx = global_idx.to(self._device).long()
        boxes_data = self.scene["boxes"].data
        
        # Use reshape (safe for non-contiguous) instead of view
        pos = boxes_data.object_pos_w.reshape(-1, 3)[global_idx]
        quat = boxes_data.object_quat_w.reshape(-1, 4)[global_idx]
        
        return pos, quat
    
    def _setup_scene(self):
        """
        Configure stage-level scene objects (Isaac Lab 5.0 API).

        IsaacLab 5.0 API update: Assets (pallet, boxes) are now defined in
        PalletSceneCfg and automatically loaded when InteractiveScene(cfg)
        is created. The scene.add() method has been removed.

        This method handles:
        1. Ground plane spawning
        2. Container prim creation for Boxes/Pallet (required by RigidObjectCollection)

        Downstream code expects `self.scene["boxes"]` and `self.scene["pallet"]`
        which are automatically available from the scene config.
        """
        import omni.usd
        stage = omni.usd.get_context().get_stage()

        def is_prim_path_valid(path: str) -> bool:
            try:
                prim = stage.GetPrimAtPath(path)
                return prim.IsValid()
            except Exception:
                return False
            
        # Ground plane with cement-gray appearance (collision enabled by default).
        # Version-safe: not all IsaacLab versions accept "visual_material" in
        # GroundPlaneCfg.  Try it first; fall back to "color" or plain.
        _ground_kwargs: dict = {}
        try:
            _ground_kwargs["visual_material"] = PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.53, 0.50),  # warm concrete gray
                roughness=0.92,
                metallic=0.0,
            )
            _ground_cfg = GroundPlaneCfg(**_ground_kwargs)
        except TypeError:
            # Older/newer IsaacLab without visual_material support
            _ground_kwargs.pop("visual_material", None)
            # Some versions expose a simple "color" field instead
            _has_color = hasattr(GroundPlaneCfg, "color") or "color" in {
                f.name for f in getattr(GroundPlaneCfg, "__dataclass_fields__", {}).values()
            }
            if _has_color:
                _ground_kwargs["color"] = (0.55, 0.53, 0.50)
            _ground_cfg = GroundPlaneCfg(**_ground_kwargs)
            print("[INFO] GroundPlaneCfg: visual_material unsupported, using fallback")

        spawn_ground_plane(
            "/World/groundPlane",
            _ground_cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )
        
        # =====================================================================
        # Stage Lighting for Headless Rendering
        # =====================================================================
        # Without lighting, RTX renders are completely black even with cameras.
        # Add DomeLight (ambient fill) and DistantLight (directional key light).
        
        # DomeLight: high-intensity ambient fill
        light_path = "/World/DomeLight"
        if not _is_prim_path_valid(light_path):
            _create_prim(
                light_path,
                "DomeLight",
                attributes={
                    "inputs:intensity": 3000.0,
                    "inputs:color": (1.0, 1.0, 1.0),
                },
            )
            print("[INFO] Created DomeLight at /World/DomeLight for headless rendering")
        
        # DistantLight: directional key light for shadows and depth perception
        dist_light_path = "/World/DistantLight"
        if not _is_prim_path_valid(dist_light_path):
            _create_prim(
                dist_light_path,
                "DistantLight",
                attributes={
                    "inputs:intensity": 5000.0,
                    "inputs:color": (1.0, 0.98, 0.95),
                    "inputs:angle": 1.0,
                },
            )
            print("[INFO] Created DistantLight at /World/DistantLight for headless rendering")

        # =====================================================================
        # Visual-Only Concrete / Linoleum Floor
        # =====================================================================
        # Thin visual slab placed just below z=0 to cover the default grid
        # ground plane. Collision stays on the ground plane itself.
        floor_path = "/World/FloorVisual"
        if self.cfg.floor_visual_enabled and not _is_prim_path_valid(floor_path):
            _fsx, _fsy = self.cfg.floor_size_xy
            _ft = self.cfg.floor_thickness
            _fc = self.cfg.floor_color
            _fz = -_ft / 2.0 - 0.001  # top surface at z=-0.001 (avoids z-fight with ground plane)
            _floor_spawned = False
            # Strategy 1: CuboidCfg with visual_material only (no physics)
            try:
                floor_spawner = CuboidCfg(
                    size=(_fsx, _fsy, _ft),
                    visual_material=PreviewSurfaceCfg(
                        diffuse_color=_fc,
                        roughness=0.9,
                        metallic=0.0,
                    ),
                )
                floor_spawner.func(
                    floor_path, floor_spawner,
                    translation=(0.0, 0.0, _fz),
                    orientation=(1.0, 0.0, 0.0, 0.0),
                )
                _floor_spawned = True
                print(f"[INFO] Spawned visual floor slab via CuboidCfg "
                      f"({_fsx}x{_fsy}x{_ft}, color={_fc})")
            except Exception as e:
                print(f"[INFO] CuboidCfg visual-only floor failed ({e}), trying USD fallback")
            
            # Strategy 2: raw USD Cube prim + PreviewSurface shader
            if not _floor_spawned:
                try:
                    from pxr import UsdGeom, UsdShade, Sdf, Gf
                    stage = _get_stage()
                    cube_prim = _create_prim(floor_path, "Cube")
                    cube_prim.GetAttribute("size").Set(1.0)
                    xform = UsdGeom.Xformable(cube_prim)
                    xform.ClearXformOpOrder()
                    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, _fz))
                    xform.AddScaleOp().Set(Gf.Vec3f(_fsx, _fsy, _ft))
                    # Apply concrete shader
                    mat_path = f"{floor_path}/ConcreteMat"
                    mat = UsdShade.Material.Define(stage, mat_path)
                    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
                    shader.CreateIdAttr("UsdPreviewSurface")
                    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                        Gf.Vec3f(*_fc)
                    )
                    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
                    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                    mat.CreateSurfaceOutput().ConnectToSource(
                        shader.ConnectableAPI(), "surface"
                    )
                    UsdShade.MaterialBindingAPI.Apply(cube_prim).Bind(mat)
                    print(f"[INFO] Spawned visual floor slab via USD fallback "
                          f"({_fsx}x{_fsy}x{_ft}, color={_fc})")
                except Exception as e2:
                    print(f"[WARNING] Both floor strategies failed: {e2}")

        # IsaacLab 5.0: Create container Xform prims for rigid object collections.
        # RigidObjectCollection expects parent prims to exist before spawning.
        # We create them under the source env path (env_0) which is then cloned.
        source_env_path = self.scene.env_prim_paths[0]
        boxes_path = f"{source_env_path}/Boxes"

        if not _is_prim_path_valid(boxes_path):
            _create_prim(boxes_path, "Xform")
        
        # =====================================================================
        # Optional: Visual Pallet Mesh (STL→USD)
        # =====================================================================
        # When enabled, converts STL to USD (cached), spawns as visual-only prim
        # at the pallet position. The cuboid pallet collider remains for physics.
        if self.cfg.use_pallet_mesh_visual:
            self._spawn_pallet_mesh_visual(source_env_path)
        
        # =====================================================================
        # Mockup-Mode Physics Overrides
        # =====================================================================
        if self.cfg.mockup_mode:
            self._apply_mockup_physics(source_env_path)
    
    def _spawn_pallet_mesh_visual(self, source_env_path: str):
        """
        Spawn visual-only pallet mesh from STL file with auto-centering.
        
        Pipeline:
          1. Resolve STL path (robust glob fallback)
          2. Convert STL→USD via MeshConverter (cached)
          3. Spawn at origin
          4. Compute world bbox via UsdGeom.BBoxCache
          5. Auto-center XY on pallet collider + align Z base to collider top
          6. Apply wood material + hide cuboid pallet visual
        
        Args:
            source_env_path: USD path to env_0 (e.g. "/World/envs/env_0")
        """
        import os
        import glob
        import hashlib
        
        # --- 1. Resolve STL path ---
        stl_path = self.cfg.pallet_mesh_stl_path
        if not os.path.isabs(stl_path):
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            stl_path = os.path.join(os.path.dirname(pkg_dir), stl_path)
        
        if not os.path.exists(stl_path):
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            assets_dir = os.path.join(os.path.dirname(pkg_dir), "assets")
            stl_files = sorted(glob.glob(os.path.join(assets_dir, "*.stl")) +
                               glob.glob(os.path.join(assets_dir, "*.STL")))
            if stl_files:
                pallet_files = [f for f in stl_files if "pallet" in os.path.basename(f).lower()]
                stl_path = pallet_files[0] if pallet_files else stl_files[0]
                print(f"[INFO] Auto-selected pallet STL: {os.path.basename(stl_path)}")
            else:
                print(f"[WARNING] No STL files found in {assets_dir}, skipping pallet mesh")
                return
        print(f"[INFO] Using pallet STL: {stl_path}")
        
        # --- 2. Convert STL→USD (cached) ---
        cache_dir = self.cfg.pallet_mesh_cache_dir
        if not os.path.isabs(cache_dir):
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir = os.path.join(os.path.dirname(pkg_dir), cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        
        stl_hash = hashlib.md5(open(stl_path, "rb").read()).hexdigest()[:8]
        scale_str = "_".join(f"{s:.4f}" for s in self.cfg.pallet_mesh_scale)
        usd_name = f"pallet_mesh_{stl_hash}_{scale_str}.usd"
        usd_path = os.path.join(cache_dir, usd_name)
        
        if not os.path.exists(usd_path):
            try:
                from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
                converter_cfg = MeshConverterCfg(
                    asset_path=stl_path,
                    usd_dir=cache_dir,
                    usd_file_name=usd_name,
                    force_usd_conversion=False,
                    make_instanceable=False,
                )
                converter = MeshConverter(converter_cfg)
                usd_path = converter.usd_path
                print(f"[INFO] Converted pallet mesh STL→USD: {usd_path}")
            except Exception as e:
                print(f"[WARNING] Failed to convert pallet mesh: {e}")
                return
        else:
            print(f"[INFO] Using cached pallet mesh USD: {usd_path}")
        
        # --- 3. Spawn at origin first (will be repositioned in step 5) ---
        mesh_prim_path = f"{source_env_path}/PalletMeshVisual"
        try:
            from isaaclab.sim.spawners.from_files import UsdFileCfg
            spawner = UsdFileCfg(
                usd_path=usd_path,
                scale=self.cfg.pallet_mesh_scale,
                rigid_props=None,
                collision_props=None,
            )
            spawner.func(
                mesh_prim_path, spawner,
                translation=(0.0, 0.0, 0.0),
                orientation=self.cfg.pallet_mesh_offset_quat_wxyz,
            )
            print(f"[INFO] Spawned visual pallet mesh at {mesh_prim_path}")
        except Exception as e:
            print(f"[WARNING] Failed to spawn pallet mesh visual: {e}")
            return
        
        # --- 4. Compute bounding box for auto-centering ---
        # Read pallet collider transform from USD (robust, no hardcoding)
        pallet_prim_path = f"{source_env_path}/Pallet"
        dx, dy, dz = 0.0, 0.0, 0.0
        try:
            from pxr import UsdGeom, Usd, Gf
            stage = _get_stage()
            
            # --- 4a. Read pallet collider pose from USD ---
            pallet_prim = stage.GetPrimAtPath(pallet_prim_path)
            collider_center = Gf.Vec3d(0.0, 0.0, 0.075)  # fallback
            pallet_half_h = 0.075  # half of 0.15m cuboid height
            if pallet_prim.IsValid():
                pallet_xform = UsdGeom.Xformable(pallet_prim)
                local_mat = pallet_xform.GetLocalTransformation()
                collider_center = local_mat.ExtractTranslation()
                # Read cuboid half-height from scene config
                pallet_size_z = 0.15  # from PalletSceneCfg pallet CuboidCfg
                pallet_half_h = pallet_size_z / 2.0
                print(f"[INFO] Pallet collider center from USD: "
                      f"({collider_center[0]:.4f}, {collider_center[1]:.4f}, {collider_center[2]:.4f})")
            else:
                print("[WARNING] Pallet prim not found, using fallback center (0,0,0.075)")
            
            collider_bottom_z = collider_center[2] - pallet_half_h  # = 0.0
            
            # --- 4b. Compute mesh bounding box ---
            mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
            if mesh_prim.IsValid():
                cache = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(),
                    includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                    useExtentsHint=False,  # False = recompute from geometry (STL extents hints are unreliable)
                )
                bbox = cache.ComputeWorldBound(mesh_prim)
                aligned = bbox.ComputeAlignedRange()
                min_pt = aligned.GetMin()
                max_pt = aligned.GetMax()
                center = (min_pt + max_pt) / 2.0
                
                print(f"[INFO] Pallet mesh bbox:")
                print(f"  min = ({min_pt[0]:.4f}, {min_pt[1]:.4f}, {min_pt[2]:.4f})")
                print(f"  max = ({max_pt[0]:.4f}, {max_pt[1]:.4f}, {max_pt[2]:.4f})")
                print(f"  center = ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
                
                # --- 5a. Auto-center XY on pallet collider center ---
                if self.cfg.pallet_mesh_auto_center:
                    dx = collider_center[0] - center[0]
                    dy = collider_center[1] - center[1]
                
                # --- 5b. Auto-align Z: mesh bottom → collider bottom (z=0) ---
                if self.cfg.pallet_mesh_auto_align_z:
                    dz = collider_bottom_z - min_pt[2]
                
                print(f"[INFO] Auto-correction: dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")
                
                # Fallback: if bottom is still floating, force it down
                corrected_min_z = min_pt[2] + dz
                if corrected_min_z > 0.005:
                    extra_dz = -corrected_min_z
                    dz += extra_dz
                    print(f"[WARNING] Mesh still floating (min_z={corrected_min_z:.4f}), "
                          f"applying extra dz={extra_dz:.4f}")
            else:
                print("[WARNING] Pallet mesh prim not valid for bbox computation")
        except Exception as e:
            print(f"[WARNING] BBox computation failed, using manual offset: {e}")
        
        # --- 5c. Apply correction + user offset ---
        user_off = self.cfg.pallet_mesh_offset_pos
        final_x = dx + user_off[0]
        final_y = dy + user_off[1]
        final_z = dz + user_off[2]
        
        try:
            from pxr import UsdGeom, Gf
            stage = _get_stage()
            mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
            if mesh_prim.IsValid():
                xformable = UsdGeom.Xformable(mesh_prim)
                # Find or create translate op
                ops = xformable.GetOrderedXformOps()
                translate_op = None
                for op in ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break
                if translate_op is None:
                    translate_op = xformable.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(final_x, final_y, final_z))
                print(f"[INFO] Pallet mesh final position: ({final_x:.4f}, {final_y:.4f}, {final_z:.4f})")
        except Exception as e:
            print(f"[WARNING] Could not reposition pallet mesh: {e}")
        
        # --- 6a. Apply wood material ---
        try:
            from pxr import UsdShade, Sdf, Gf
            stage = _get_stage()
            mat_path = f"{mesh_prim_path}/WoodMaterial"
            mat = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(0.72, 0.55, 0.35)  # warm wood tone
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
            if mesh_prim.IsValid():
                UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(mat)
            print("[INFO] Applied wood material to pallet mesh")
        except Exception as e:
            print(f"[WARNING] Could not apply wood material: {e}")
        
        # --- 6b. Hide cuboid pallet visual (keep collision) ---
        pallet_visual_path = f"{source_env_path}/Pallet"
        try:
            from pxr import UsdGeom
            stage = _get_stage()
            pallet_prim = stage.GetPrimAtPath(pallet_visual_path)
            if pallet_prim.IsValid():
                imageable = UsdGeom.Imageable(pallet_prim)
                imageable.MakeInvisible()
                print(f"[INFO] Hidden cuboid pallet visual at {pallet_visual_path}")
        except Exception as e:
            print(f"[WARNING] Could not hide cuboid pallet: {e}")
    
    def _apply_mockup_physics(self, source_env_path: str):
        """
        Apply mockup-mode physics overrides to box prims via USD attributes.
        
        Sets high friction, zero restitution, linear/angular damping,
        solver iterations, and velocity clamping on all box prims.
        Only called when mockup_mode=True; does not affect training defaults.
        
        Args:
            source_env_path: USD path to env_0 (e.g. "/World/envs/env_0")
        """
        try:
            from pxr import UsdPhysics, PhysxSchema, Sdf
            stage = _get_stage()
            if not stage: return
            cfg = self.cfg
            
            # --- Pallet collider verification (debug log) ---
            pallet_path = f"{source_env_path}/Pallet"
            pallet_prim = stage.GetPrimAtPath(pallet_path)
            if pallet_prim.IsValid():
                has_collision = pallet_prim.HasAPI(UsdPhysics.CollisionAPI)
                has_rb = pallet_prim.HasAPI(UsdPhysics.RigidBodyAPI)
                is_kinematic = False
                if has_rb:
                    rb = UsdPhysics.RigidBodyAPI(pallet_prim)
                    kin_attr = rb.GetKinematicEnabledAttr()
                    is_kinematic = kin_attr.Get() if kin_attr else False
                # Read pose
                from pxr import UsdGeom, Gf
                xformable = UsdGeom.Xformable(pallet_prim)
                local_mat = xformable.GetLocalTransformation()
                center = local_mat.ExtractTranslation()
                pallet_bottom_z = center[2] - 0.075  # half of 0.15m cuboid
                print(f"[INFO] Pallet collider verification:")
                print(f"  path={pallet_path}")
                print(f"  CollisionAPI={has_collision}, RigidBodyAPI={has_rb}, kinematic={is_kinematic}")
                print(f"  center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}), bottom_z={pallet_bottom_z:.4f}")
            else:
                print(f"[WARNING] Pallet prim not found at {pallet_path}")
            
            applied_count = 0
            for i in range(cfg.max_boxes):
                box_path = f"{source_env_path}/Boxes/box_{i}"
                box_prim = stage.GetPrimAtPath(box_path)
                if not box_prim.IsValid():
                    continue
                
                # --- Physics material (friction + restitution) ---
                mat_path = f"{box_path}/MockupPhysMat"
                UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(Sdf.Path(mat_path)))
                mat_prim = stage.GetPrimAtPath(mat_path)
                mat_api = UsdPhysics.MaterialAPI(mat_prim)
                mat_api.CreateStaticFrictionAttr().Set(cfg.mockup_box_static_friction)
                mat_api.CreateDynamicFrictionAttr().Set(cfg.mockup_box_dynamic_friction)
                mat_api.CreateRestitutionAttr().Set(cfg.mockup_box_restitution)
                
                # Bind material to box
                UsdPhysics.MaterialAPI.Apply(box_prim)
                phys_mat = UsdPhysics.MaterialAPI(box_prim)
                phys_mat.CreateStaticFrictionAttr().Set(cfg.mockup_box_static_friction)
                phys_mat.CreateDynamicFrictionAttr().Set(cfg.mockup_box_dynamic_friction)
                phys_mat.CreateRestitutionAttr().Set(cfg.mockup_box_restitution)
                
                # --- Rigid body stability (damping, velocity clamp, solver, depenet, CCD) ---
                rb_api = None
                if box_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    rb_api = PhysxSchema.PhysxRigidBodyAPI(box_prim)
                else:
                    rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(box_prim)
                
                if rb_api is not None:
                    rb_api.CreateLinearDampingAttr().Set(cfg.mockup_box_linear_damping)
                    rb_api.CreateAngularDampingAttr().Set(cfg.mockup_box_angular_damping)
                    rb_api.CreateMaxLinearVelocityAttr().Set(cfg.mockup_box_max_linear_velocity)
                    rb_api.CreateMaxAngularVelocityAttr().Set(cfg.mockup_box_max_angular_velocity)
                    rb_api.CreateSolverPositionIterationCountAttr().Set(cfg.mockup_solver_position_iterations)
                    rb_api.CreateSolverVelocityIterationCountAttr().Set(cfg.mockup_solver_velocity_iterations)
                    rb_api.CreateMaxDepenetrationVelocityAttr().Set(cfg.mockup_max_depenetration_velocity)
                    if cfg.mockup_enable_ccd:
                        rb_api.CreateEnableCCDAttr().Set(True)
                
                # Collision properties: contact/rest offsets
                col_api = None
                if box_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                    col_api = PhysxSchema.PhysxCollisionAPI(box_prim)
                else:
                    col_api = PhysxSchema.PhysxCollisionAPI.Apply(box_prim)
                if col_api is not None:
                    col_api.CreateContactOffsetAttr().Set(cfg.mockup_contact_offset)
                    col_api.CreateRestOffsetAttr().Set(cfg.mockup_rest_offset)
                
                applied_count += 1
            
            # --- Apply friction to pallet surface for box-pallet contact ---
            pallet_path = f"{source_env_path}/Pallet"
            pallet_prim = stage.GetPrimAtPath(pallet_path)
            if pallet_prim.IsValid():
                UsdPhysics.MaterialAPI.Apply(pallet_prim)
                pal_mat = UsdPhysics.MaterialAPI(pallet_prim)
                pal_mat.CreateStaticFrictionAttr().Set(cfg.mockup_pallet_static_friction)
                pal_mat.CreateDynamicFrictionAttr().Set(cfg.mockup_pallet_dynamic_friction)
                pal_mat.CreateRestitutionAttr().Set(0.0)
                print(f"[INFO] Applied mockup friction to pallet collider: "
                      f"static={cfg.mockup_pallet_static_friction}, "
                      f"dynamic={cfg.mockup_pallet_dynamic_friction}")
            
            print(f"[INFO] Applied mockup physics to {applied_count} box prims")
            print(f"  friction: static={cfg.mockup_box_static_friction}, "
                  f"dynamic={cfg.mockup_box_dynamic_friction}, "
                  f"restitution={cfg.mockup_box_restitution}")
            print(f"  damping: linear={cfg.mockup_box_linear_damping}, "
                  f"angular={cfg.mockup_box_angular_damping}")
            print(f"  velocity clamp: linear={cfg.mockup_box_max_linear_velocity} m/s, "
                  f"angular={cfg.mockup_box_max_angular_velocity} rad/s")
            print(f"  depenetration: max_vel={cfg.mockup_max_depenetration_velocity} m/s, "
                  f"CCD={cfg.mockup_enable_ccd}")
            print(f"  solver iterations: pos={cfg.mockup_solver_position_iterations}, "
                  f"vel={cfg.mockup_solver_velocity_iterations}")
            
        except ImportError as e:
            print(f"[WARNING] PhysX/USD schemas not available for mockup physics: {e}")
        except Exception as e:
            print(f"[WARNING] Failed to apply mockup physics: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """
        Construct observations (GPU-only, no numpy).
        
        Returns:
            dict: {"policy": (N, obs_dim), "critic": (N, obs_dim)}
        """
        n = self.num_envs
        device = torch.device(self._device)
        
        # 1. Get box poses from scene (GPU tensors)
        M = self.cfg.max_boxes   # always 50, fixed tensor size
        K = self.cfg.num_boxes   # active boxes (<= M)
        if "boxes" in self.scene.keys():
            boxes_data = self.scene["boxes"].data
            all_pos = boxes_data.object_pos_w.reshape(-1, 3)   # (num_envs*num_boxes, 3)
            all_rot_wxyz = boxes_data.object_quat_w.reshape(-1, 4)  # (num_envs*num_boxes, 4) (w,x,y,z)
            
            box_pos = all_pos.view(n, M, 3)
            # Convert Isaac (w,x,y,z) → Warp (x,y,z,w) before rasterization
            box_rot = wxyz_to_xyzw(all_rot_wxyz).view(n, M, 4)
        else:
            # Fallback for testing
            box_pos = torch.zeros(n, M, 3, device=device)
            box_rot = torch.zeros(n, M, 4, device=device)
            box_rot[:, :, 0] = 1.0  # Identity quat w=1
        
        # Debug: log tensor dimensions (once every 200 obs steps)
        if not hasattr(self, '_obs_dbg_count'):
            self._obs_dbg_count = 0
        self._obs_dbg_count += 1
        if self._obs_dbg_count <= 1 or self._obs_dbg_count % 200 == 0:
            _active_k = min(K, self.box_idx.max().item() + 1) if K < M else M
            print(f"  [OBS DBG] M={M} K={K} all_pos.numel={all_pos.numel()} "
                  f"box_pos.shape={list(box_pos.shape)} "
                  f"active_placed~{_active_k} step={self._obs_dbg_count}")
        
        # Pallet positions (centered at origin per env)
        pallet_pos = torch.zeros(n, 3, device=device)
        
        # 2. Generate heightmap
        if self.cfg.heightmap_source == "depth_camera" and self._depth_converter is not None:
            # ---------------------------------------------------------------
            # Depth camera path: read sensor → convert → heightmap
            # ---------------------------------------------------------------
            depth_cam = self.scene["depth_camera"]
            depth_data = depth_cam.data
            
            # Get depth image: Isaac Lab returns (N, H, W, 1) for single-channel
            depth_img = depth_data.output["distance_to_image_plane"]  # (N, H, W, 1)
            if depth_img.dim() == 4 and depth_img.shape[-1] == 1:
                depth_img = depth_img.squeeze(-1)  # (N, H, W)
            
            # Camera world poses
            cam_pos = depth_data.pos_w  # (N, 3)
            cam_quat_wxyz = depth_data.quat_w_world  # (N, 4) wxyz
            
            # Decimation: reuse cached heightmap on non-update steps
            self._depth_step_count += 1
            dec = self.cfg.depth_cam_decimation
            if dec > 1 and self._cached_depth_heightmap is not None:
                if (self._depth_step_count - 1) % dec != 0:
                    heightmap = self._cached_depth_heightmap
                else:
                    heightmap = self._depth_converter.depth_to_heightmap(
                        depth_img, cam_pos, cam_quat_wxyz
                    )
                    self._cached_depth_heightmap = heightmap
            else:
                heightmap = self._depth_converter.depth_to_heightmap(
                    depth_img, cam_pos, cam_quat_wxyz
                )
                self._cached_depth_heightmap = heightmap
            
            # Optional debug frame saving
            if self.cfg.depth_debug_save_frames:
                import os
                os.makedirs(self.cfg.depth_debug_save_dir, exist_ok=True)
                step = self._depth_step_count
                torch.save(
                    {"depth": depth_img[0].cpu(), "heightmap": heightmap[0].cpu()},
                    os.path.join(self.cfg.depth_debug_save_dir, f"frame_{step:06d}.pt"),
                )
        else:
            # ---------------------------------------------------------------
            # Warp path (default): analytical rasterization from box poses
            # ---------------------------------------------------------------
            # Only boxes that have been placed (indices 0..box_idx-1) are "active" for
            # rasterization. Use strict < to exclude the current/next box.
            box_indices = torch.arange(self.cfg.max_boxes, device=device).view(1, -1)
            active_mask = box_indices < self.box_idx.view(-1, 1)  # (N, max_boxes)
            
            # Update preallocated buffer: copy active dims, zero inactive
            self._box_dims_for_hmap.copy_(self.box_dims)
            self._box_dims_for_hmap[~active_mask] = 0.0
            
            # Move inactive boxes off-map (guarantees they're never in rasterization bounds)
            box_pos_for_hmap = box_pos.clone()
            box_pos_for_hmap[~active_mask] = self._inactive_box_pos

            heightmap = self.heightmap_gen.forward(
                box_pos_for_hmap.reshape(-1, 3),
                box_rot.reshape(-1, 4),
                self._box_dims_for_hmap.reshape(-1, 3),
                pallet_pos,
            )  # (N, H, W)
        
        # 3. Normalize and flatten heightmap
        heightmap_norm = heightmap / self.cfg.max_height
        heightmap_flat = heightmap_norm.view(n, -1)  # (N, H*W)
        
        # Cache raw heightmap for action masking (height constraint)
        self._last_heightmap = heightmap  # (N, H, W) in meters
        
        # 4. Buffer state (N, slots*features)
        buffer_flat = self.buffer_state.view(n, -1)
        
        # 5. Current box dimensions
        idx = self.box_idx.clamp(0, self.cfg.max_boxes - 1)
        env_idx = torch.arange(n, device=device)
        current_dims = self.box_dims[env_idx, idx]  # (N, 3)
        
        # 6. Payload and mass observations
        # Normalize payload: current on-pallet mass / max allowed
        payload_norm = (self.payload_kg / self.cfg.max_payload_kg).unsqueeze(-1)  # (N, 1)
        # Normalize current box mass
        max_box_mass = self.cfg.base_box_mass_kg + self.cfg.box_mass_variance
        current_box_mass = self.box_mass_kg[env_idx, idx]  # (N,)
        current_mass_norm = (current_box_mass / max_box_mass).unsqueeze(-1)  # (N, 1)
        
        # 7. Constraint observations (for future domain randomization)
        # These are currently constant but visible to policy for future DR support
        max_payload_norm = torch.full((n, 1), self.cfg.max_payload_kg / 1000.0, device=device)  # Normalize to ~0.5
        max_stack_height_norm = torch.full((n, 1), self.cfg.max_stack_height / 3.0, device=device)  # Normalize to ~0.6
        
        # 8. Proprioception (placeholder - would come from robot)
        proprio = torch.zeros(n, self.cfg.robot_state_dim, device=device)
        
        # 9. Concatenate all observations
        obs = torch.cat([
            heightmap_flat,        # (N, 38400)
            buffer_flat,           # (N, 60)
            current_dims,          # (N, 3)
            payload_norm,          # (N, 1)
            current_mass_norm,     # (N, 1)
            max_payload_norm,      # (N, 1) - for future domain randomization
            max_stack_height_norm, # (N, 1) - for future domain randomization
            proprio,               # (N, 24)
        ], dim=-1)
        
        # Shape/device assertion (fast, only runs in debug or once per step)
        # IsaacLab 5.x: num_observations can be None; set on first call
        if self.cfg.num_observations is None:
            self.cfg.num_observations = int(obs.shape[-1])
        expected_obs_dim = int(self.cfg.num_observations)
        assert obs.shape == (n, expected_obs_dim), \
            f"Obs shape {obs.shape} != expected ({n}, {expected_obs_dim})"
        assert obs.device == device, \
            f"Obs device {obs.device} != expected {device}"
        
        return {"policy": obs, "critic": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """
        Compute rewards (pure PyTorch, JIT-compatible).
        Also computes task KPIs logged via self.extras for TensorBoard.
        
        Includes:
        - Height constraint penalty (invalid placement attempt)
        - Settling stability rewards (drift, fall, stable)
        - Infeasible payload penalty
        - Original placement success/failure rewards
        
        Returns:
            rewards: (N,) tensor
        """
        n = self.num_envs
        device = self._device
        
        rewards = torch.zeros(n, device=device)
        
        # Initialize KPI tracking tensors (per-env booleans)
        place_success = torch.zeros(n, dtype=torch.bool, device=device)
        place_failure = torch.zeros(n, dtype=torch.bool, device=device)
        retrieve_success = torch.zeros(n, dtype=torch.bool, device=device)
        
        # =====================================================================
        # Penalty for height-invalid actions (attempted but blocked)
        # =====================================================================
        rewards = rewards + self.cfg.reward_invalid_height * self._height_invalid_mask.float()
        
        # =====================================================================
        # Penalty for infeasible state (handled in _get_dones but reward here)
        # =====================================================================
        rewards = rewards + self.cfg.reward_infeasible * self._infeasible_mask.float()
        if self._infeasible_mask.any():
            self._kpi_infeasible_count += self._infeasible_mask.float().sum()
        
        # Penalty for storing (using buffer)
        rewards = rewards - 0.1 * self.store_mask.float()
        
        # Bonus for successful retrieve
        rewards = rewards + 2.0 * self.valid_retrieve.float()
        
        # Buffer age penalty
        ages = self.buffer_state[:, :, 4].sum(dim=1)  # Sum of ages
        rewards = rewards - 0.01 * ages
        
        # Success/failure for placement
        # Only evaluate envs where a placement occurred (last_moved_box_id >= 0)
        valid_eval = self.last_moved_box_id >= 0
        
        if "boxes" in self.scene.keys() and valid_eval.any():
            # Optimization: only read box state for valid envs
            valid_envs = valid_eval.nonzero(as_tuple=False).flatten()
            eval_box_idx = self.last_moved_box_id[valid_envs]
            global_idx = valid_envs * self.cfg.max_boxes + eval_box_idx
            current_pos, current_quat = self._get_box_pos_quat(global_idx)
            
            target_pos = self.last_target_pos[valid_envs]
            target_quat = self.last_target_quat[valid_envs]
            
            # XY distance
            dist = torch.norm(current_pos[:, :2] - target_pos[:, :2], dim=-1)
            fell = current_pos[:, 2] < 0.05
            
            # Use configured thresholds for consistency with termination and settling
            # XY drift check
            unstable_xy = dist > self.cfg.drift_xy_threshold
            # Rotation drift check using numerically-stable quaternion angular distance
            rot_error_deg = quat_angle_deg(current_quat, target_quat)
            unstable_rot = rot_error_deg > self.cfg.drift_rot_threshold
            # Combined unstable: either XY or rotation exceeded threshold
            unstable = unstable_xy | unstable_rot
            
            # Track rotation-only unstable for KPI (not XY, only rotation)
            rot_only_unstable = unstable_rot & ~unstable_xy & ~fell
            self._kpi_unstable_rot_count += rot_only_unstable.float().sum()
            
            # Compute failure/success for valid envs only
            failure_valid = fell | unstable
            success_valid = ~failure_valid
            
            # Expand back to full env tensors
            failure = torch.zeros(n, dtype=torch.bool, device=device)
            success = torch.zeros(n, dtype=torch.bool, device=device)
            failure[valid_envs] = failure_valid
            success[valid_envs] = success_valid
            
            # Apply active_place_mask (PLACE or RETRIEVE that results in placement)
            failure = failure & self.active_place_mask
            success = success & self.active_place_mask
            
            rewards = rewards - 10.0 * failure.float()
            rewards = rewards + 1.0 * success.float()
            
            # Volume bonus for successful placement
            dims = self.box_dims[valid_envs, eval_box_idx]
            vol = dims[:, 0] * dims[:, 1] * dims[:, 2]
            vol_rewards = torch.zeros(n, device=device)
            vol_rewards[valid_envs] = vol * success_valid.float()
            rewards = rewards + vol_rewards
            
            # Track KPIs: separate PLACE from RETRIEVE
            place_only = (self.active_place_mask & ~self.valid_retrieve)
            place_success = success & place_only
            place_failure = failure & place_only
            retrieve_success = success & self.valid_retrieve
        
        # =====================================================================
        # Queue KPI evaluations with settling window
        # =====================================================================
        # For PLACE: queue the placed box for later KPI evaluation
        place_only = (self.active_place_mask & ~self.valid_retrieve)
        if place_only.any():
            place_envs = place_only.nonzero(as_tuple=False).flatten()
            # +1 to account for decrement happening in same frame
            self._kpi_countdown[place_envs] = self.cfg.kpi_settle_steps + 1
            self._kpi_pending_type[place_envs] = 1  # 1 = place
            self._kpi_pending_box_id[place_envs] = self.last_moved_box_id[place_envs]
            self._kpi_pending_target[place_envs] = self.last_target_pos[place_envs]
            self._kpi_pending_target_quat[place_envs] = self.last_target_quat[place_envs]
        
        # For valid RETRIEVE: queue the retrieved box for later KPI evaluation
        if self.valid_retrieve.any():
            retr_envs = self.valid_retrieve.nonzero(as_tuple=False).flatten()
            # +1 to account for decrement happening in same frame
            self._kpi_countdown[retr_envs] = self.cfg.kpi_settle_steps + 1
            self._kpi_pending_type[retr_envs] = 2  # 2 = retrieve
            self._kpi_pending_box_id[retr_envs] = self.last_moved_box_id[retr_envs]
            self._kpi_pending_target[retr_envs] = self.last_target_pos[retr_envs]
            self._kpi_pending_target_quat[retr_envs] = self.last_target_quat[retr_envs]
        
        # =====================================================================
        # Evaluate settled KPIs (countdown reached 0)
        # =====================================================================
        # Decrement countdown for envs with pending evaluations
        pending = self._kpi_countdown > 0
        self._kpi_countdown[pending] -= 1
        
        # Find envs that just reached 0 (ready for evaluation)
        ready_mask = (self._kpi_countdown == 0) & (self._kpi_pending_type > 0)
        
        if "boxes" in self.scene.keys() and ready_mask.any():
            ready_envs = ready_mask.nonzero(as_tuple=False).flatten()
            eval_box_ids = self._kpi_pending_box_id[ready_envs]
            eval_targets = self._kpi_pending_target[ready_envs]
            eval_target_quats = self._kpi_pending_target_quat[ready_envs]
            eval_types = self._kpi_pending_type[ready_envs]
            
            # Get settled box positions and orientations
            global_idx = ready_envs * self.cfg.max_boxes + eval_box_ids
            settled_pos, settled_quat = self._get_box_pos_quat(global_idx)
            
            # Evaluate success criteria using configured thresholds
            dist = torch.norm(settled_pos[:, :2] - eval_targets[:, :2], dim=-1)
            fell = settled_pos[:, 2] < 0.05
            # XY drift check
            unstable_xy = dist > self.cfg.drift_xy_threshold
            # Rotation drift check
            rot_error_deg = quat_angle_deg(settled_quat, eval_target_quats)
            unstable_rot = rot_error_deg > self.cfg.drift_rot_threshold
            # Combined unstable
            unstable = unstable_xy | unstable_rot
            success = ~(fell | unstable)
            
            # Update running accumulators
            place_mask = (eval_types == 1)
            retr_mask = (eval_types == 2)
            
            self._kpi_place_success_count += (success & place_mask).float().sum()
            self._kpi_place_fail_count += (~success & place_mask).float().sum()
            self._kpi_retrieve_success_count += (success & retr_mask).float().sum()
            self._kpi_retrieve_fail_count += (~success & retr_mask).float().sum()
            self._kpi_eval_count += len(ready_envs)
            
            # Clear pending state
            self._kpi_pending_type[ready_envs] = 0
            self._kpi_pending_box_id[ready_envs] = -1
        
        # =====================================================================
        # Settling Stability Evaluation (separate from KPI settling)
        # =====================================================================
        # Decrement settling countdown
        settling = self._settle_countdown > 0
        self._settle_countdown[settling] -= 1
        
        # Check for falls during settling (any settling box z < 0.05)
        if "boxes" in self.scene.keys() and settling.any():
            settling_envs = settling.nonzero(as_tuple=False).flatten()
            box_ids = self._settle_box_id[settling_envs]
            valid_box_mask = box_ids >= 0
            if valid_box_mask.any():
                valid_settling_envs = settling_envs[valid_box_mask]
                valid_box_ids = box_ids[valid_box_mask]
                global_idx = valid_settling_envs * self.cfg.max_boxes + valid_box_ids
                current_pos, _ = self._get_box_pos_quat(global_idx)
                fell = current_pos[:, 2] < 0.05
                
                # Collapse during settling -> terminate and penalize
                if fell.any():
                    collapse_envs = valid_settling_envs[fell]
                    rewards[collapse_envs] += self.cfg.reward_fall
                    self._settle_countdown[collapse_envs] = 0
                    self._settle_box_id[collapse_envs] = -1
                    self._kpi_collapse_count += fell.float().sum()
        
        # Evaluate completed settling (countdown just reached 0 with valid box)
        done_settling = (self._settle_countdown == 0) & (self._settle_box_id >= 0)
        if "boxes" in self.scene.keys() and done_settling.any():
            done_envs = done_settling.nonzero(as_tuple=False).flatten()
            box_ids = self._settle_box_id[done_envs]
            global_idx = done_envs * self.cfg.max_boxes + box_ids
            
            current_pos, current_quat = self._get_box_pos_quat(global_idx)
            target_pos = self._settle_target_pos[done_envs]
            target_quat = self._settle_target_quat[done_envs]
            
            # Compute XY drift
            drift_xy = torch.norm(current_pos[:, :2] - target_pos[:, :2], dim=-1)
            
            # Rotation drift: compute angle from quaternion dot product
            # Both quats are (w,x,y,z) format
            quat_dot = (current_quat * target_quat).sum(dim=-1).abs()
            drift_rot_rad = 2 * torch.acos(quat_dot.clamp(-1 + 1e-7, 1 - 1e-7))
            drift_rot_deg = drift_rot_rad * 180.0 / 3.14159265359
            
            # Check if fell
            fell = current_pos[:, 2] < 0.05
            
            # Determine outcome and apply rewards
            exceeded_drift = (drift_xy > self.cfg.drift_xy_threshold) | (drift_rot_deg > self.cfg.drift_rot_threshold)
            
            # Fall -> reward_fall (already handled during settling, but check end state too)
            fell_envs = done_envs[fell]
            if len(fell_envs) > 0:
                rewards[fell_envs] += self.cfg.reward_fall
                self._kpi_collapse_count += fell.float().sum()
            
            # Drifted but didn't fall -> reward_drift
            drifted_mask = exceeded_drift & ~fell
            drifted_envs = done_envs[drifted_mask]
            if len(drifted_envs) > 0:
                rewards[drifted_envs] += self.cfg.reward_drift
                self._kpi_drift_count += drifted_mask.float().sum()
            
            # Stable (no drift, no fall) -> reward_stable
            stable_mask = ~exceeded_drift & ~fell
            stable_envs = done_envs[stable_mask]
            if len(stable_envs) > 0:
                rewards[stable_envs] += self.cfg.reward_stable
                self._kpi_stable_count += stable_mask.float().sum()
            
            # Accumulate drift metrics for averaging
            self._kpi_total_drift_xy += drift_xy.sum()
            self._kpi_total_drift_deg += drift_rot_deg.sum()
            self._kpi_settle_eval_count += len(done_envs)
            
            # Clear settling state
            self._settle_box_id[done_envs] = -1
        
        # Accumulate payload for utilization metric
        self._kpi_total_payload += self.payload_kg.sum()
        
        # =====================================================================
        # Compute and log task KPIs (CPU scalars for TensorBoard)
        # =====================================================================
        # Use settled accumulators for accurate KPIs
        total_place = self._kpi_place_success_count + self._kpi_place_fail_count
        total_retr = self._kpi_retrieve_success_count + self._kpi_retrieve_fail_count
        total_settle = self._kpi_settle_eval_count + 1e-8
        
        # Compute rates (avoid div by zero)
        place_success_rate = self._kpi_place_success_count / (total_place + 1e-8)
        place_failure_rate = self._kpi_place_fail_count / (total_place + 1e-8)
        retrieve_success_rate = self._kpi_retrieve_success_count / (total_retr + 1e-8)
        
        # Store accept rate: use true valid_store (includes empty-slot condition)
        store_attempts = self.store_mask.float().sum()
        store_accept_rate = self.valid_store.float().sum() / (store_attempts + 1e-8)
        
        # Buffer occupancy (immediate metric)
        buffer_occupancy = self.buffer_has_box.float().mean()
        
        # New constraint KPIs
        drift_rate = self._kpi_drift_count / total_settle
        collapse_rate = self._kpi_collapse_count / total_settle
        infeasible_rate = self._kpi_infeasible_count / (self._kpi_eval_count + 1e-8)
        avg_drift_xy = self._kpi_total_drift_xy / total_settle
        avg_drift_deg = self._kpi_total_drift_deg / total_settle
        payload_utilization = self.payload_kg.mean() / self.cfg.max_payload_kg
        
        # Log to extras with CPU conversion at the boundary
        # KPIs are emitted via self.extras and forwarded by RslRlVecEnvWrapper
        self.extras["metrics/place_success_rate"] = place_success_rate.detach().cpu().item()
        self.extras["metrics/place_failure_rate"] = place_failure_rate.detach().cpu().item()
        self.extras["metrics/retrieve_success_rate"] = retrieve_success_rate.detach().cpu().item()
        self.extras["metrics/store_accept_rate"] = store_accept_rate.detach().cpu().item()
        self.extras["metrics/buffer_occupancy"] = buffer_occupancy.detach().cpu().item()
        
        # New constraint KPIs
        self.extras["metrics/drift_rate"] = drift_rate.detach().cpu().item()
        self.extras["metrics/collapse_rate"] = collapse_rate.detach().cpu().item()
        self.extras["metrics/infeasible_rate"] = infeasible_rate.detach().cpu().item()
        self.extras["metrics/avg_drift_xy"] = avg_drift_xy.detach().cpu().item()
        self.extras["metrics/avg_drift_deg"] = avg_drift_deg.detach().cpu().item()
        self.extras["metrics/payload_utilization"] = payload_utilization.detach().cpu().item()
        # Rotation-only unstable rate (immediate evaluation)
        unstable_rot_rate = self._kpi_unstable_rot_count / (self._kpi_eval_count + 1e-8)
        self.extras["metrics/unstable_rot_rate"] = unstable_rot_rate.detach().cpu().item()
        
        return rewards
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute termination and truncation flags.
        
        Termination conditions:
        - Box fell or became unstable after placement
        - All boxes used
        - Payload infeasibility (prospective mass exceeds max)
        
        Returns:
            terminated: (N,) bool tensor
            truncated: (N,) bool tensor
        """
        n = self.num_envs
        device = self._device
        
        terminated = torch.zeros(n, dtype=torch.bool, device=device)
        truncated = torch.zeros(n, dtype=torch.bool, device=device)
        
        # Terminate if box fell or all boxes placed
        # Only evaluate envs where a placement occurred (last_moved_box_id >= 0)
        valid_eval = self.last_moved_box_id >= 0
        
        if "boxes" in self.scene.keys() and valid_eval.any():
            # =====================================================================
            # TERMINATION LOGIC ALIGNMENT FIX
            # =====================================================================
            # Previously used hardcoded thresholds (0.10m, no rotation check).
            # Now uses cfg.drift_xy_threshold and cfg.drift_rot_threshold to be
            # consistent with the settling stability evaluation in _get_rewards().
            #
            # This ensures termination and settling KPIs agree on what is "failed".
            # =====================================================================
            
            # Optimization: only read box state for valid envs
            valid_envs = valid_eval.nonzero(as_tuple=False).flatten()
            eval_box_idx = self.last_moved_box_id[valid_envs]
            global_idx = valid_envs * self.cfg.max_boxes + eval_box_idx
            current_pos, current_quat = self._get_box_pos_quat(global_idx)
            
            target_pos = self.last_target_pos[valid_envs]
            dist = torch.norm(current_pos[:, :2] - target_pos[:, :2], dim=-1)
            
            # Target rotation was saved for most recent place (or identity if not set)
            target_quat = self._settle_target_quat[valid_envs]
            
            # XY distance check
            fell = current_pos[:, 2] < 0.05
            drift_xy = dist
            
            # Rotation drift check using quaternion dot product
            quat_dot = (current_quat * target_quat).sum(dim=-1).abs()
            drift_rot_rad = 2 * torch.acos(quat_dot.clamp(-1 + 1e-7, 1 - 1e-7))
            drift_rot_deg = drift_rot_rad * (180.0 / 3.14159265359)
            
            # Use configured thresholds (same as settling evaluation)
            exceeded_xy = drift_xy > self.cfg.drift_xy_threshold
            exceeded_rot = drift_rot_deg > self.cfg.drift_rot_threshold
            unstable = exceeded_xy | exceeded_rot
            
            # Compute failure for valid envs, expand to full tensor
            failure_valid = fell | unstable
            active_place_valid = self.active_place_mask[valid_envs]
            
            # Only terminate if valid placement occurred and it failed
            terminated[valid_envs] = active_place_valid & failure_valid
        
        # Check if all boxes used (box_idx already incremented on Place, so >= means done)
        terminated = terminated | (self.box_idx >= self.cfg.max_boxes)
        
        # =====================================================================
        # Payload Infeasibility Termination
        # =====================================================================
        # Compute remaining mass to place (conservative estimate using base mass)
        remaining_boxes = (self.cfg.max_boxes - self.box_idx).float()
        remaining_mass = remaining_boxes * self.cfg.base_box_mass_kg
        
        # Buffer mass (mass stored in buffer slots)
        buffer_mass = (self.buffer_state[:, :, 5] * self.buffer_has_box.float()).sum(dim=1)
        
        # Total prospective mass if all remaining boxes were placed
        prospective_total = self.payload_kg + buffer_mass + remaining_mass
        
        # Infeasible if prospective total exceeds max
        self._infeasible_mask = prospective_total > self.cfg.max_payload_kg
        terminated = terminated | self._infeasible_mask
        
        # Truncation from time limit is handled by DirectRLEnv base class.
        # We intentionally return all-False here; the base class ORs in the
        # timeout truncation after calling _get_dones().
        # truncated stays all-False
        
        return terminated, truncated
    
    def _reset_idx(self, env_ids: torch.Tensor):
        """
        Partial reset for specified environments (tensor-sliced).
        
        Args:
            env_ids: Tensor of environment indices to reset
        """
        super()._reset_idx(env_ids)
        
        if len(env_ids) == 0:
            return
        
        device = self._device
        
        # Reset state tensors (tensor-sliced assignment)
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
        
        # Reset KPI settling state
        self._kpi_countdown[env_ids] = 0
        self._kpi_pending_type[env_ids] = 0
        self._kpi_pending_box_id[env_ids] = -1
        self._kpi_pending_target[env_ids] = 0.0
        
        # =====================================================================
        # Reset Mass / Payload State
        # =====================================================================
        self.payload_kg[env_ids] = 0.0
        
        # Randomize box masses for new episode
        base_mass = self.cfg.base_box_mass_kg
        variance = self.cfg.box_mass_variance
        rand_mass = torch.rand(len(env_ids), self.cfg.max_boxes, device=device)
        self.box_mass_kg[env_ids] = base_mass + (rand_mass * 2 - 1) * variance
        
        # =====================================================================
        # Reset Settling Stability State
        # =====================================================================
        self._settle_countdown[env_ids] = 0
        self._settle_box_id[env_ids] = -1
        self._settle_target_pos[env_ids] = 0.0
        self._settle_target_quat[env_ids] = 0.0
        self._settle_target_quat[env_ids, 0] = 1.0  # Identity quaternion w component
        
        # =====================================================================
        # Reset Height Constraint State
        # =====================================================================
        self._height_invalid_mask[env_ids] = False
        
        # =====================================================================
        # Reset Infeasibility State
        # =====================================================================
        self._infeasible_mask[env_ids] = False
        
        # Randomize box dimensions for new episode
        base_dims = torch.tensor([0.4, 0.3, 0.2], device=device)
        rand_offset = torch.rand(len(env_ids), self.cfg.max_boxes, 3, device=device) * 0.2 - 0.1
        self.box_dims[env_ids] = base_dims + rand_offset
        
        # Move all boxes off-map initially (they'll be placed when actions are taken)
        # This ensures inactive boxes never pollute the heightmap
        if "boxes" in self.scene.keys():
            num_reset = len(env_ids)
            max_b = self.cfg.max_boxes
            
            # Build pose tensor: (num_reset_envs, max_boxes, 7) = pos(3) + quat(4)
            inactive_pos = self._inactive_box_pos.unsqueeze(0).unsqueeze(0).expand(num_reset, max_b, 3)
            inactive_quat = torch.zeros(num_reset, max_b, 4, device=device)
            inactive_quat[..., 0] = 1.0  # Identity quaternion (w,x,y,z)
            
            # Concatenate to pose: (num_reset_envs, max_boxes, 7)
            inactive_pose = torch.cat([inactive_pos, inactive_quat], dim=-1)
            
            # RigidObjectCollection API: write_object_pose_to_sim(pose, env_ids)
            # pose shape: (num_envs, num_objects, 7)
            self.scene["boxes"].write_object_pose_to_sim(inactive_pose, env_ids=env_ids)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Called by DirectRLEnv.step() before physics stepping.
        
        Isaac Lab API: This method receives actions and stores them in a buffer.
        The framework then calls _apply_action() (no args) to apply them.
        
        Handles two action formats (backwards-compatible):
        1. Discrete indices from MultiCategorical policy: values 0..K-1 (ints or floats >1)
           → Converted to center-of-bin normalized values in [-1, 1]
        2. Already-normalized float actions in [-1, 1] (legacy)
           → Passed through with clamp
        
        Args:
            actions: Actions from policy (N, 5) — discrete indices or normalized floats
        """
        actions = actions.to(self._device).float()
        
        # =====================================================================
        # Discrete Index Detection & Conversion
        # =====================================================================
        # The policy (PalletizerActorCritic.act()) outputs integer indices for
        # each action dimension: [op∈{0..2}, slot∈{0..9}, x∈{0..15}, y∈{0..23}, rot∈{0..1}].
        # Previously, clamp(-1,1) would corrupt indices >1 (e.g. slot=9 → 1.0).
        #
        # Detection heuristic: if ANY value > 1.5, actions are discrete indices.
        # Center-of-bin: norm_val = (idx + 0.5) / K * 2.0 - 1.0
        is_discrete = actions.abs().max() > 1.5
        
        if is_discrete:
            dims = self.cfg.action_dims  # (3, 10, 16, 24, 2)
            for col, k in enumerate(dims):
                actions[:, col] = (actions[:, col].float() + 0.5) / k * 2.0 - 1.0
        
        self._actions = torch.clamp(actions, -1.0, 1.0)
    
    def _apply_action(self) -> None:
        """
        Apply continuous action by decoding to discrete/continuous sub-actions.
        
        Isaac Lab API: Called by DirectRLEnv.step() after _pre_physics_step().
        Reads actions from self._actions (set by _pre_physics_step).
        
        Action decoding from continuous [-1, 1] to original intent:
        - a[0] -> op_type: discrete {0,1,2} (PLACE, STORE, RETRIEVE)
        - a[1] -> slot_idx: discrete {0..9}
        - a[2] -> target_x: continuous [-0.6, +0.6] meters
        - a[3] -> target_y: continuous [-0.4, +0.4] meters
        - a[4] -> rot_idx: discrete {0,1} (0° or 90°)
        
        Includes:
        - Height constraint validation
        - Payload mass updates
        - Settling window arm
        """
        n = self.num_envs
        device = self._device
        action = self._actions  # (N, 5) continuous in [-1, 1]
        
        # =====================================================================
        # Decode continuous actions to discrete/continuous sub-actions
        # =====================================================================
        # Helper to map [-1,1] -> {0..K-1}
        def to_discrete(a: torch.Tensor, k: int) -> torch.Tensor:
            # Map [-1,1] to [0,1] then scale to [0,K)
            return torch.floor(((a + 1.0) * 0.5) * k).long().clamp(0, k - 1)
        
        # Parse action components
        # Discrete: op_type (3 values), slot_idx (10 values), rot_idx (2 values)
        op_type = to_discrete(action[:, 0], self.cfg.action_dims[0])  # 3 ops
        slot_idx = to_discrete(action[:, 1], self.cfg.action_dims[1])  # 10 slots
        rot_idx = to_discrete(action[:, 4], self.cfg.action_dims[4])  # 2 rotations
        
        # Continuous: x, y positions mapped to workspace bounds
        # x: [-1,1] -> [-half_x, +half_x] where half_x = pallet_size[0]/2
        # y: [-1,1] -> [-half_y, +half_y] where half_y = pallet_size[1]/2
        half_x = self.cfg.pallet_size[0] / 2.0  # 0.6m
        half_y = self.cfg.pallet_size[1] / 2.0  # 0.4m
        target_x = action[:, 2] * half_x  # [-0.6, +0.6]
        target_y = action[:, 3] * half_y  # [-0.4, +0.4]
        
        # For backward compatibility with grid-based code, compute grid indices
        # grid_x: 0..15, grid_y: 0..23
        num_x = self.cfg.action_dims[2]  # 16
        num_y = self.cfg.action_dims[3]  # 24
        grid_x = to_discrete(action[:, 2], num_x)
        grid_y = to_discrete(action[:, 3], num_y)
        
        # =====================================================================
        # Compute target position from action grid
        # =====================================================================
        # GEOMETRY FIX: Action grid now correctly aligns with pallet_size.
        # pallet_size = (1.2, 0.8) means X spans 1.2m, Y spans 0.8m.
        #
        # grid_x: 0..15 (16 cells) → covers 1.2m → step_x = 1.2/16 = 0.075m
        # grid_y: 0..23 (24 cells) → covers 0.8m → step_y = 0.8/24 = 0.0333m
        #
        # Pallet is centered at origin, so:
        #   X ranges from -0.6 to +0.6 (half_x = 0.6)
        #   Y ranges from -0.4 to +0.4 (half_y = 0.4)
        # =====================================================================
        pallet_x = self.cfg.pallet_size[0]  # 1.2m
        pallet_y = self.cfg.pallet_size[1]  # 0.8m
        num_x = self.cfg.action_dims[2]  # 16
        num_y = self.cfg.action_dims[3]  # 24
        
        step_x = pallet_x / num_x  # 0.075m per cell
        step_y = pallet_y / num_y  # 0.0333m per cell
        half_x = pallet_x / 2.0    # 0.6m
        half_y = pallet_y / 2.0    # 0.4m
        
        # Map grid index to world coordinates (cell center)
        target_x = grid_x.float() * step_x - half_x + step_x / 2
        target_y = grid_y.float() * step_y - half_y + step_y / 2
        target_z = torch.full((n,), 1.5, device=device)  # Drop height
        
        # Operation masks
        # PLACE (0) and RETRIEVE (2) both result in a box being on the pallet after physics
        # but ONLY PLACE should move box_idx here; RETRIEVE moves its physical box in _handle_buffer_actions
        self.active_place_mask = (op_type == 0) | (op_type == 2)
        self.store_mask = (op_type == 1)
        self.retrieve_mask = (op_type == 2)
        
        # =====================================================================
        # Height Constraint Validation (FIX: per-op box height)
        # =====================================================================
        # CRITICAL FIX: Previously used box_dims[box_idx] for ALL operations.
        # This was wrong for RETRIEVE, which must use the retrieved box's height
        # from buffer_state, not the current fresh box height.
        #
        # Per-op height logic:
        # - PLACE (op=0): uses box_dims[box_idx] (the next fresh box)
        # - RETRIEVE (op=2): uses buffer_state[slot_idx, 2] (height stored in buffer)
        # - STORE (op=1): no height check (not placing a box on pallet)
        
        # Reset height invalid mask
        self._height_invalid_mask[:] = False
        
        # Only PLACE and RETRIEVE need height checks (they put a box on the pallet)
        place_mask = (op_type == 0)
        retrieve_mask_local = (op_type == 2)
        needs_height_check = place_mask | retrieve_mask_local
        
        if self._last_heightmap is not None and needs_height_check.any():
            # Map grid coordinates to heightmap pixels
            # grid_x: 0..15 -> pixel_x: 0..239  (spans 0.8m on X axis)
            # grid_y: 0..23 -> pixel_y: 0..159  (spans 1.2m on Y axis)
            pixel_x = (grid_x.float() / max(1, self.cfg.action_dims[2] - 1) * (self.cfg.map_shape[1] - 1)).long()
            pixel_y = (grid_y.float() / max(1, self.cfg.action_dims[3] - 1) * (self.cfg.map_shape[0] - 1)).long()
            
            # Clamp to valid range
            pixel_x = pixel_x.clamp(0, self.cfg.map_shape[1] - 1)
            pixel_y = pixel_y.clamp(0, self.cfg.map_shape[0] - 1)
            
            # Sample local height at target position
            env_idx = torch.arange(n, device=device)
            local_height = self._last_heightmap[env_idx, pixel_y, pixel_x]  # (N,)
            
            # Compute per-op box height (PLACE vs RETRIEVE)
            # PLACE: use fresh box height from box_dims
            box_idx_clamped = self.box_idx.clamp(0, self.cfg.max_boxes - 1)
            place_box_height = self.box_dims[env_idx, box_idx_clamped, 2]  # (N,)
            
            # RETRIEVE: use height stored in buffer_state for the selected slot
            # buffer_state[:, :, 2] contains the H (height) dimension
            slot_idx_clamped = slot_idx.clamp(0, self.cfg.buffer_slots - 1)
            retrieve_box_height = self.buffer_state[env_idx, slot_idx_clamped, 2]  # (N,)
            
            # Select correct height per-op: PLACE uses fresh box, RETRIEVE uses buffer box
            box_height = torch.where(place_mask, place_box_height, retrieve_box_height)
            
            # Predict top of stack after placement
            predicted_top = local_height + box_height
            
            # Mark as invalid if exceeds max stack height (only for ops that need it)
            height_exceeds = predicted_top > self.cfg.max_stack_height
            self._height_invalid_mask = height_exceeds & needs_height_check
        
        # Mask for PLACE-only envs (for writing box_idx pose)
        # Exclude height-invalid actions from actual placement
        place_only_mask = (op_type == 0) & ~self._height_invalid_mask
        
        # Build target position tensor
        target_pos = torch.stack([target_x, target_y, target_z], dim=-1)
        
        # Rotation quaternion
        quat = torch.zeros(n, 4, device=device)
        quat[:, 0] = 1.0  # Identity (w, x, y, z)
        rot_mask = (rot_idx == 1)
        quat[rot_mask, 0] = 0.7071068  # 90° Z rotation
        quat[rot_mask, 3] = 0.7071068
        
        # Save target for stability check (used for all placement types including retrieve)
        self.last_target_pos[:, 0] = target_x
        self.last_target_pos[:, 1] = target_y
        self.last_target_pos[:, 2] = 0.0
        # Save target quaternion for rotation drift evaluation
        self.last_target_quat.copy_(quat)
        
        # Apply to simulation: ONLY move box_idx for PLACE actions
        # STORE and RETRIEVE handle their own box movements in _handle_buffer_actions
        if "boxes" in self.scene.keys() and place_only_mask.any():
            place_envs = place_only_mask.nonzero(as_tuple=False).flatten()
            box_ids = self.box_idx[place_envs]
            
            # Build pose: (num_place_envs, 7) = pos(3) + quat(4)
            pose = torch.cat([target_pos[place_envs], quat[place_envs]], dim=-1)
            
            # Build velocity: (num_place_envs, 6) = lin_vel(3) + ang_vel(3)
            vel = torch.zeros(len(place_envs), 6, device=device)
            
            # RigidObjectCollection API: write per-object state
            # We need to write state for specific (env, object) pairs
            # Use write_object_state_to_sim with reshaped tensors
            num_place = len(place_envs)
            # Reshape to (num_envs, 1, ...) for single object per env
            pose_reshaped = pose.unsqueeze(1)  # (num_place, 1, 7)
            vel_reshaped = vel.unsqueeze(1)    # (num_place, 1, 6)
            
            # global flat index (env, box)
            flat_idx = place_envs * self.cfg.max_boxes + box_ids

            boxes_data = self.scene["boxes"].data

            pos  = boxes_data.object_pos_w       # (E, B, 3)
            quat_w = boxes_data.object_quat_w    # (E, B, 4)
            lin  = boxes_data.object_lin_vel_w   # (E, B, 3)
            ang  = boxes_data.object_ang_vel_w   # (E, B, 3)

            B = pos.shape[1]

            flat_idx = flat_idx.to(torch.long)
            env_ids = flat_idx // B
            box_ids2 = flat_idx %  B

            pos[env_ids, box_ids2, :] = target_pos[place_envs]
            quat_w[env_ids, box_ids2, :] = quat[place_envs]
            lin[env_ids, box_ids2, :] = 0.0
            ang[env_ids, box_ids2, :] = 0.0

            self.scene["boxes"].write_data_to_sim()

            
            # =====================================================================
            # Payload Update for PLACE
            # =====================================================================
            placed_box_mass = self.box_mass_kg[place_envs, self.box_idx[place_envs]]
            self.payload_kg[place_envs] += placed_box_mass
            
            # =====================================================================
            # Arm Settling Window for PLACE
            # =====================================================================
            self._settle_countdown[place_envs] = self.cfg.settle_steps
            self._settle_box_id[place_envs] = self.box_idx[place_envs]
            self._settle_target_pos[place_envs] = self.last_target_pos[place_envs]
            self._settle_target_quat[place_envs] = quat[place_envs]
        
        # Buffer logic for store/retrieve
        self._handle_buffer_actions(action)
    
    def _handle_buffer_actions(self, action: torch.Tensor):
        """
        Handle store and retrieve buffer operations with physical box tracking.
        
        Physical buffer semantics:
        - STORE parks an existing physical box in a holding area and records its ID
        - RETRIEVE moves that same parked physical box back onto the pallet
        - The buffer does NOT create new boxes
        
        Also handles:
        - Mass tracking in buffer_state[:, :, 5]
        - Payload updates for STORE (remove from pallet) and RETRIEVE (add to pallet)
        - Settling window arm for RETRIEVE
        
        Example scenario:
          Place box A (box_idx=0) -> Store slot0 -> Place box B (box_idx=1)
          -> Retrieve slot0 should return box A (physical ID 0), NOT box B
        """
        device = self._device
        n = self.num_envs
        env_idx = torch.arange(n, device=device)
        
        slot_idx = action[:, 1].long()
        op_type = action[:, 0]
        
        # Reset last_moved_box_id; will be set appropriately below
        self.last_moved_box_id[:] = -1
        
        # =======================================================================
        # STORE: Park the last-placed physical box in a buffer slot
        # =======================================================================
        # Validation:
        # - box_idx must be > 0 (there must be a placed box to store)
        # - slot must not be occupied (Option A: reject store into occupied slot)
        has_box_to_store = self.box_idx > 0
        slot_is_empty = ~self.buffer_has_box[env_idx, slot_idx]
        valid_store = self.store_mask & has_box_to_store & slot_is_empty
        self.valid_store = valid_store  # Store for KPI logging
        
        if valid_store.any():
            store_envs = valid_store.nonzero(as_tuple=False).flatten()
            store_slots = slot_idx[store_envs]
            
            # The physical box being stored is the last placed: box_idx - 1
            stored_physical_id = (self.box_idx[store_envs] - 1).clamp(0, self.cfg.max_boxes - 1)
            dims = self.box_dims[store_envs, stored_physical_id]
            stored_mass = self.box_mass_kg[store_envs, stored_physical_id]
            
            # Update buffer_state (dims, active flag, age, mass)
            self.buffer_state[store_envs, store_slots, :3] = dims
            self.buffer_state[store_envs, store_slots, 3] = 1.0  # Active flag
            self.buffer_state[store_envs, store_slots, 4] = 0.0  # Reset age
            self.buffer_state[store_envs, store_slots, 5] = stored_mass  # Store mass
            
            # Track physical box identity in slot
            self.buffer_has_box[store_envs, store_slots] = True
            self.buffer_box_id[store_envs, store_slots] = stored_physical_id
            
            # ===================================================================
            # Payload Update for STORE: remove stored box mass from pallet
            # ===================================================================
            self.payload_kg[store_envs] -= stored_mass
            
            # Move the stored box off-map (to holding area)
            if "boxes" in self.scene.keys():
                global_store_idx = (store_envs * self.cfg.max_boxes + stored_physical_id).long()
                holding_pos = self._inactive_box_pos.expand(len(store_envs), 3)
                holding_quat = torch.zeros(len(store_envs), 4, device=device)
                holding_quat[:, 0] = 1.0  # Identity quaternion
                
                # Isaac Lab API: use data buffer + write_data_to_sim (not write_root_pose_to_sim)
                boxes_data = self.scene["boxes"].data
                boxes_data.object_pos_w.reshape(-1, 3)[global_store_idx] = holding_pos
                boxes_data.object_quat_w.reshape(-1, 4)[global_store_idx] = holding_quat
                boxes_data.object_lin_vel_w.reshape(-1, 3)[global_store_idx] = 0.0
                boxes_data.object_ang_vel_w.reshape(-1, 3)[global_store_idx] = 0.0
                self.scene["boxes"].write_data_to_sim()
            
            # STORE does not count as a placement; last_moved_box_id stays -1 for these envs
        
        # =======================================================================
        # RETRIEVE: Move a parked physical box back onto the pallet
        # =======================================================================
        # Also check height constraint for RETRIEVE
        retrieve_height_valid = ~self._height_invalid_mask  # Already computed in _apply_action
        has_box_in_slot = self.buffer_has_box[env_idx, slot_idx]
        self.valid_retrieve = self.retrieve_mask & has_box_in_slot & retrieve_height_valid
        
        if self.valid_retrieve.any():
            retr_envs = self.valid_retrieve.nonzero(as_tuple=False).flatten()
            retr_slots = slot_idx[retr_envs]
            
            # Get the physical box ID from the buffer slot
            retrieved_physical_id = self.buffer_box_id[retr_envs, retr_slots]
            retrieved_mass = self.buffer_state[retr_envs, retr_slots, 5]
            
            # Move THAT physical box to the target position
            # Note: target position was already computed in _apply_action
            if "boxes" in self.scene.keys():
                global_retr_idx = retr_envs * self.cfg.max_boxes + retrieved_physical_id
                
                # Build target pose for retrieved boxes (using corrected geometry)
                pallet_x = self.cfg.pallet_size[0]  # 1.2m
                pallet_y = self.cfg.pallet_size[1]  # 0.8m
                num_x = self.cfg.action_dims[2]  # 16
                num_y = self.cfg.action_dims[3]  # 24
                
                step_x = pallet_x / num_x  # 0.075m per cell
                step_y = pallet_y / num_y  # 0.0333m per cell
                half_x = pallet_x / 2.0    # 0.6m
                half_y = pallet_y / 2.0    # 0.4m
                
                grid_x = action[retr_envs, 2]
                grid_y = action[retr_envs, 3]
                rot_idx = action[retr_envs, 4]
                
                target_x = grid_x.float() * step_x - half_x + step_x / 2
                target_y = grid_y.float() * step_y - half_y + step_y / 2
                target_z = torch.full((len(retr_envs),), 1.5, device=device)
                target_pos = torch.stack([target_x, target_y, target_z], dim=-1)
                
                quat = torch.zeros(len(retr_envs), 4, device=device)
                quat[:, 0] = 1.0
                rot_mask = (rot_idx == 1)
                quat[rot_mask, 0] = 0.7071068
                quat[rot_mask, 3] = 0.7071068
                
                # Write to sim using data buffer approach (RigidObjectCollection API)
                boxes_data = self.scene["boxes"].data
                boxes_data.object_pos_w.view(-1, 3)[global_retr_idx] = target_pos
                boxes_data.object_quat_w.view(-1, 4)[global_retr_idx] = quat
                boxes_data.object_lin_vel_w.view(-1, 3)[global_retr_idx] = 0.0
                boxes_data.object_ang_vel_w.view(-1, 3)[global_retr_idx] = 0.0
                
                self.scene["boxes"].write_data_to_sim()
                
                # ===============================================================
                # Arm Settling Window for RETRIEVE
                # ===============================================================
                self._settle_countdown[retr_envs] = self.cfg.settle_steps
                self._settle_box_id[retr_envs] = retrieved_physical_id
                # Save target pos (ground level for stability check)
                settle_target = target_pos.clone()
                settle_target[:, 2] = 0.0  # Ground level for XY drift check
                self._settle_target_pos[retr_envs] = settle_target
                self._settle_target_quat[retr_envs] = quat
            
            # ===================================================================
            # Payload Update for RETRIEVE: add retrieved box mass to pallet
            # ===================================================================
            self.payload_kg[retr_envs] += retrieved_mass
            
            # Clear buffer slot
            self.buffer_has_box[retr_envs, retr_slots] = False
            self.buffer_box_id[retr_envs, retr_slots] = -1
            self.buffer_state[retr_envs, retr_slots] = 0.0
            
            # Record last_moved_box_id for reward/done evaluation
            self.last_moved_box_id[retr_envs] = retrieved_physical_id
            
            # RETRIEVE counts as a placement
            self.active_place_mask = self.active_place_mask | self.valid_retrieve
        
        # Age all buffer slots
        self.buffer_state[:, :, 4] += 1.0
        
        # =======================================================================
        # PLACE: Advance box_idx and record last_moved_box_id
        # =======================================================================
        # PLACE (op=0): consumes a new box -> increment box_idx
        # STORE (op=1): parks existing box, does NOT consume -> no increment
        # RETRIEVE (op=2): moves parked box, does NOT consume new box -> no increment
        # Only increment for valid PLACE (not height-invalid)
        place_mask = (op_type == 0) & ~self._height_invalid_mask
        
        # For PLACE, last_moved_box_id = box_idx (the box being placed)
        # Note: we set this BEFORE incrementing box_idx
        self.last_moved_box_id = torch.where(
            place_mask,
            self.box_idx,  # Current box_idx is the placed box
            self.last_moved_box_id
        )
        
        self.box_idx += place_mask.long()
    
    # NOTE: step() is NOT overridden — DirectRLEnv.step() handles the lifecycle:
    #   1. _pre_physics_step(action)
    #   2. _apply_action(action)  ← our logic, including box_idx increment at end
    #   3. Physics stepping (cfg.decimation × sim.step())
    #   4. _post_physics_step()
    #   5. _get_observations() / _get_rewards() / _get_dones()
    #   6. Reset handling and episode_length_buf updates


    def get_action_mask(self) -> torch.Tensor:
        """
        Return an action mask over the flattened MultiDiscrete logits.

        Shape: (num_envs, sum(action_dims)), dtype=bool, device matches env device.

        Implements:
        - Height-based X/Y masking: ONLY for PLACE operation (uses known fresh box height)
        - Buffer slot masking: marks empty slots (informational for policy)
        
        IMPORTANT FIX: Height masking is NOT applied for RETRIEVE operations because:
        - The MultiDiscrete action space selects [Op, Slot, X, Y, Rot] simultaneously
        - For RETRIEVE, the box height depends on which slot is selected
        - We cannot mask X/Y based on height when the slot (and thus box height) is unknown
        - The actual height validation happens in _apply_action after the action is taken
        
        This mask is compatible with PalletizerActorCritic.act() and evaluate()
        methods, which expect (N, total_logits) bool tensors.
        """
        n = self.num_envs
        device = self._device
        total_logits = sum(self.cfg.action_dims)
        
        # Start with all actions valid
        mask = torch.ones(n, total_logits, dtype=torch.bool, device=device)
        
        # Logit offsets for each dimension:
        # Op: 0..2 (indices 0-2)
        # Slot: 3..12 (indices 3-12)
        # X: 13..28 (indices 13-28)
        # Y: 29..52 (indices 29-52)
        # Rot: 53..54 (indices 53-54)
        op_start = 0
        slot_start = self.cfg.action_dims[0]  # 3
        x_start = slot_start + self.cfg.action_dims[1]  # 13
        y_start = x_start + self.cfg.action_dims[2]  # 29
        rot_start = y_start + self.cfg.action_dims[3]  # 53
        
        # =====================================================================
        # Height-based Grid Masking (PLACE ONLY)
        # =====================================================================
        # FIX: Apply height masking ONLY for PLACE operations.
        # For RETRIEVE, the box height depends on the selected slot, which is
        # chosen simultaneously with X/Y in the MultiDiscrete action space.
        # We cannot accurately mask X/Y positions for RETRIEVE without knowing
        # which slot will be selected. The actual height validation for RETRIEVE
        # happens in _apply_action where the full action is known.
        #
        # Note: This is a limitation of MultiDiscrete action spaces. The policy
        # should learn to avoid height violations through reward signals.
        if self._last_heightmap is not None:
            heightmap = self._last_heightmap  # (N, H, W)
            
            # Get grid dimensions
            num_x = self.cfg.action_dims[2]  # 16
            num_y = self.cfg.action_dims[3]  # 24
            
            # Map grid coordinates to heightmap pixels
            grid_xs = torch.arange(num_x, device=device)
            grid_ys = torch.arange(num_y, device=device)
            
            pixel_xs = (grid_xs.float() / max(1, num_x - 1) * (self.cfg.map_shape[1] - 1)).long()
            pixel_ys = (grid_ys.float() / max(1, num_y - 1) * (self.cfg.map_shape[0] - 1)).long()
            
            pixel_xs = pixel_xs.clamp(0, self.cfg.map_shape[1] - 1)
            pixel_ys = pixel_ys.clamp(0, self.cfg.map_shape[0] - 1)
            
            # Get FRESH box height per env (for PLACE only)
            # This is the box that would be placed if PLACE is selected
            idx = self.box_idx.clamp(0, self.cfg.max_boxes - 1)
            env_idx = torch.arange(n, device=device)
            fresh_box_h = self.box_dims[env_idx, idx, 2]  # (N,)
            
            # Sample heights at all grid positions: (N, num_y, num_x)
            all_heights = heightmap[:, pixel_ys[:, None], pixel_xs[None, :]]  # (N, 24, 16)
            
            # Predict top of stack after PLACE at each cell
            predicted_tops = all_heights + fresh_box_h[:, None, None]  # (N, 24, 16)
            
            # Invalid if exceeds max stack height (for PLACE operation)
            grid_invalid = predicted_tops > self.cfg.max_stack_height  # (N, 24, 16)
            
            # Mask X positions where ALL Y positions are invalid for PLACE
            all_y_invalid_at_x = grid_invalid.all(dim=1)  # (N, 16)
            mask[:, x_start:x_start + num_x] &= ~all_y_invalid_at_x
            
            # Mask Y positions where ALL X positions are invalid for PLACE
            all_x_invalid_at_y = grid_invalid.all(dim=2)  # (N, 24)
            mask[:, y_start:y_start + num_y] &= ~all_x_invalid_at_y
        
        # =====================================================================
        # Buffer Slot Masking (informational)
        # =====================================================================
        # Mark empty slots as invalid. This helps the policy learn to not
        # select RETRIEVE for empty slots. Note: STORE also uses slots but
        # targets empty slots, so this mask is primarily useful for RETRIEVE.
        # The actual validation happens in _handle_buffer_actions.
        #
        # slots: (N, buffer_slots) boolean - True if slot is occupied
        # We invert to mask empty slots (True = valid, so ~occupied for empty = invalid)
        # But since STORE wants empty slots, we leave this as informational only
        # and let the policy learn the semantics through rewards.
        
        assert mask.device == device, "Action mask device mismatch"
        return mask

