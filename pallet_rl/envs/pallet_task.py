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
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sim.spawners import shapes as shape_spawners
# IsaacLab API update: use CuboidCfg for spawning box primitives
from isaaclab.sim.spawners.shapes import CuboidCfg
# Schema configs required to spawn rigid bodies with RigidBodyAPI
from isaaclab.sim.schemas import RigidBodyPropertiesCfg, CollisionPropertiesCfg, MassPropertiesCfg
# IsaacLab API update: ground plane spawner moved to from_files module
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
# IsaacLab 5.0: prim utilities for creating container prims before spawning
from isaacsim.core.utils import prims as prim_utils

import gymnasium as gym

from pallet_rl.utils.heightmap_rasterizer import WarpHeightmapGenerator
from pallet_rl.utils.quaternions import wxyz_to_xyzw, quat_angle_deg


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
                    rigid_props=RigidBodyPropertiesCfg(),  # Dynamic rigid body
                    collision_props=CollisionPropertiesCfg(),
                    mass_props=MassPropertiesCfg(density=250.0),
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
        device="cuda:0"
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
    # Dummy gym spaces are defined here; real spaces are set at runtime in PalletTask.__init__.
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
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
        prim_utils.create_prim(
            f"{env_ns}/env_0/Boxes",
            "Xform"
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
        
        # State tensors (all GPU-resident)
        self._init_state_tensors()
        
        # Action space (gymnasium MultiDiscrete)
        self.action_space = gym.spaces.MultiDiscrete(list(self.cfg.action_dims))
        
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
        # Ground plane (simple infinite plane primitive)
        # IsaacLab API update: ground plane spawner moved to from_files module
        spawn_ground_plane(
            "/World/groundPlane",
            GroundPlaneCfg(),
            translation=(0.0, 0.0, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        # IsaacLab 5.0: Create container Xform prims for rigid object collections.
        # RigidObjectCollection expects parent prims to exist before spawning.
        # We create them under the source env path (env_0) which is then cloned.
        source_env_path = self.scene.env_prim_paths[0]
        boxes_path = f"{source_env_path}/Boxes"

        if not prim_utils.is_prim_path_valid(boxes_path):
            prim_utils.create_prim(boxes_path, "Xform")
    
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """
        Construct observations (GPU-only, no numpy).
        
        Returns:
            dict: {"policy": (N, obs_dim), "critic": (N, obs_dim)}
        """
        n = self.num_envs
        device = torch.device(self._device)
        
        # 1. Get box poses from scene (GPU tensors)
        if "boxes" in self.scene.keys():
            boxes_data = self.scene["boxes"].data
            all_pos = boxes_data.object_pos_w.reshape(-1, 3)   # (num_envs*num_boxes, 3)
            all_rot_wxyz = boxes_data.object_quat_w.reshape(-1, 4)  # (num_envs*num_boxes, 4) (w,x,y,z)
            
            box_pos = all_pos.view(n, self.cfg.max_boxes, 3)
            # Convert Isaac (w,x,y,z) → Warp (x,y,z,w) before rasterization
            box_rot = wxyz_to_xyzw(all_rot_wxyz).view(n, self.cfg.max_boxes, 4)
        else:
            # Fallback for testing
            box_pos = torch.zeros(n, self.cfg.max_boxes, 3, device=device)
            box_rot = torch.zeros(n, self.cfg.max_boxes, 4, device=device)
            box_rot[:, :, 0] = 1.0  # Identity quat w=1
        
        # Pallet positions (centered at origin per env)
        pallet_pos = torch.zeros(n, 3, device=device)
        
        # 2. Generate heightmap (GPU-only via Warp)
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
            current_pos = self.scene["boxes"].data.root_pos_w[global_idx]
            current_quat = self.scene["boxes"].data.root_quat_w[global_idx]  # (w,x,y,z)
            
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
            settled_pos = self.scene["boxes"].data.root_pos_w[global_idx]
            settled_quat = self.scene["boxes"].data.root_quat_w[global_idx]  # (w,x,y,z)
            
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
                current_pos = self.scene["boxes"].data.root_pos_w[global_idx]
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
            
            current_pos = self.scene["boxes"].data.root_pos_w[global_idx]
            current_quat = self.scene["boxes"].data.root_quat_w[global_idx]  # (w,x,y,z)
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
            current_pos = self.scene["boxes"].data.root_pos_w[global_idx]
            
            target_pos = self.last_target_pos[valid_envs]
            dist = torch.norm(current_pos[:, :2] - target_pos[:, :2], dim=-1)
            
            # Get rotation (to check rotation drift for fails like toppled boxes)
            current_quat = self.scene["boxes"].data.root_quat_w[global_idx]  # (w,x,y,z)
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
        
        IsaacLab DirectRLEnv requires this method to process actions
        before the physics simulation advances.
        
        Args:
            actions: Raw actions from the policy (N, num_actions) or (N,)
        """
        # Move actions to device
        actions = actions.to(self._device)
        
        # For MultiDiscrete: cast to long (discrete indices)
        # RSL-RL may pass float actions; our _apply_action expects integers
        if actions.dtype in (torch.float32, torch.float16, torch.bfloat16):
            actions = actions.long()
        
        # Ensure shape is (num_envs, 5) for MultiDiscrete
        if actions.dim() == 1:
            # Single action dimension - reshape
            actions = actions.view(self.num_envs, -1)
        
        # Delegate to existing action application logic
        self._apply_action(actions)
    
    def _apply_action(self, action: torch.Tensor):
        """
        Apply MultiDiscrete action.
        
        Includes:
        - Height constraint validation
        - Payload mass updates
        - Settling window arm
        
        Args:
            action: (N, 5) tensor of [Op, Slot, X, Y, Rot]
        """
        n = self.num_envs
        device = self._device
        
        # Parse action components
        op_type = action[:, 0]
        slot_idx = action[:, 1]
        grid_x = action[:, 2]
        grid_y = action[:, 3]
        rot_idx = action[:, 4]
        
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
            
            # For each placing env, set the state of its specific box
            # This requires iterating or using advanced indexing
            # Simpler approach: use set_world_poses with global indices
            global_idx = place_envs * self.cfg.max_boxes + box_ids
            
            # Get data buffer and write directly
            boxes_data = self.scene["boxes"].data
            pos = boxes_data.object_pos_w                 # (num_envs, num_boxes, 3)  (tipico)
            num_boxes = pos.shape[1]

            global_idx = global_idx.to(torch.long)
            env_ids = global_idx // num_boxes
            box_ids = global_idx %  num_boxes
            pos[env_ids, box_ids, :] = target_pos[place_envs]
            boxes_data.object_quat_w.view(-1, 4)[global_idx] = quat[place_envs]
            boxes_data.object_lin_vel_w.view(-1, 3)[global_idx] = 0.0
            boxes_data.object_ang_vel_w.view(-1, 3)[global_idx] = 0.0
            
            # Write to sim
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
                global_store_idx = store_envs * self.cfg.max_boxes + stored_physical_id
                holding_pos = self._inactive_box_pos.expand(len(store_envs), 3)
                holding_quat = torch.zeros(len(store_envs), 4, device=device)
                holding_quat[:, 0] = 1.0  # Identity quaternion
                self.scene["boxes"].write_root_pose_to_sim(
                    holding_pos, holding_quat, indices=global_store_idx
                )
            
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

