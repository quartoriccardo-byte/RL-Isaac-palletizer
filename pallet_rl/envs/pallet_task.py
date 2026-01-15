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
from typing import Dict, Any

# Isaac Lab imports (4.0+ namespace)
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sim.spawners import shapes as shape_spawners

import gymnasium as gym

from pallet_rl.utils.heightmap_rasterizer import WarpHeightmapGenerator
from pallet_rl.utils.quaternions import wxyz_to_xyzw


# =============================================================================
# Configuration
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
    
    # Scene (will be populated with robot + pallet + boxes)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=3.0
    )
    
    # Decimation (physics steps per RL step)
    # Default kept at 50 to match the previous hardcoded behaviour in step().
    decimation: int = 50
    
    # Episode length
    episode_length_s: float = 60.0
    
    # Pallet dimensions (cm -> m in code)
    pallet_size: tuple[float, float] = (1.2, 0.8)  # meters
    
    # Heightmap configuration
    map_shape: tuple[int, int] = (160, 240)  # H, W pixels
    grid_res: float = 0.005  # 0.5cm resolution
    max_height: float = 2.0  # meters (for normalization)
    
    # Box configuration
    max_boxes: int = 50
    
    # Buffer configuration
    buffer_slots: int = 10
    buffer_features: int = 5  # [L, W, H, ID, Age]
    
    # Robot state dimension
    robot_state_dim: int = 24  # 6 pos + 6 vel + gripper etc.
    
    # Observation dimension (computed)
    @property
    def num_observations(self) -> int:
        # Heightmap (flattened) + Buffer + Box dims + Proprio
        vis_dim = self.map_shape[0] * self.map_shape[1]  # 38400
        buf_dim = self.buffer_slots * self.buffer_features  # 50
        box_dim = 3
        return vis_dim + buf_dim + box_dim + self.robot_state_dim  # 38477
    
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
        obs_dim = self.cfg.num_observations
        self.observation_space = gym.spaces.Box(
            low=-float('inf'), high=float('inf'),
            shape=(obs_dim,), dtype=torch.float32
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
        
        # Target position for stability check
        self.last_target_pos = torch.zeros(n, 3, device=device)
        
        # Off-map position for inactive boxes (far away to ensure they never pollute heightmap)
        self._inactive_box_pos = torch.tensor([1e6, 1e6, -1e6], device=device)
    
    def _setup_scene(self):
        """
        Configure scene objects using Isaac Lab declarative API.

        Scene contains:
        - Ground plane (static)
        - Pallet volume (static box)
        - Boxes collection (rigid objects)

        Notes:
        - We keep this implementation minimal and declarative to avoid
          over-constraining asset paths. A production setup would replace
          the primitive pallet with a USD asset and tune physical materials.
        - Downstream code expects a `boxes` view available as
          `self.scene["boxes"]` exposing `data.root_pos_w` and
          `data.root_quat_w` tensors.
        """
        # Ground plane (simple infinite plane primitive)
        shape_spawners.spawn_ground_plane(self.scene)

        # Pallet as a simple box at the origin of each env.
        pallet_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Pallet",
            spawn=RigidObjectCfg.SpawnCfg(
                func=shape_spawners.spawn_box,
                extents=(self.cfg.pallet_size[0], self.cfg.pallet_size[1], 0.15),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.075),
                rot=(1.0, 0.0, 0.0, 0.0),  # (w,x,y,z) for Isaac scene
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
            ),
            rigid_props=RigidObjectCfg.RigidPropsCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            collision_props=RigidObjectCfg.CollisionPropsCfg(
                collision_enabled=True,
                contact_offset=0.02,
                rest_offset=0.0,
            ),
        )

        boxes_cfg = RigidObjectCollectionCfg(
            prim_path="{ENV_REGEX_NS}/Boxes",
            base_cfg=RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Boxes/box",
                spawn=RigidObjectCfg.SpawnCfg(
                    func=shape_spawners.spawn_box,
                    # Extents will be overridden at reset based on `box_dims`.
                    extents=(0.4, 0.3, 0.2),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 1.5),
                    rot=(1.0, 0.0, 0.0, 0.0),
                    lin_vel=(0.0, 0.0, 0.0),
                    ang_vel=(0.0, 0.0, 0.0),
                ),
                rigid_props=RigidObjectCfg.RigidPropsCfg(
                    disable_gravity=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                ),
                collision_props=RigidObjectCfg.CollisionPropsCfg(
                    collision_enabled=True,
                    contact_offset=0.01,
                    rest_offset=0.0,
                ),
            ),
            count_per_env=self.cfg.max_boxes,
        )

        # Register rigid objects with the interactive scene
        self.scene.add(pallet_cfg, "pallet")
        self.scene.add(boxes_cfg, "boxes")
    
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """
        Construct observations (GPU-only, no numpy).
        
        Returns:
            dict: {"policy": (N, obs_dim), "critic": (N, obs_dim)}
        """
        n = self.num_envs
        device = self._device
        
        # 1. Get box poses from scene (GPU tensors)
        if "boxes" in self.scene.keys():
            all_pos = self.scene["boxes"].data.root_pos_w  # (N*max_boxes, 3)
            all_rot_wxyz = self.scene["boxes"].data.root_quat_w  # (N*max_boxes, 4) (w,x,y,z)
            
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
        
        # 4. Buffer state (N, slots*features)
        buffer_flat = self.buffer_state.view(n, -1)
        
        # 5. Current box dimensions
        idx = self.box_idx.clamp(0, self.cfg.max_boxes - 1)
        env_idx = torch.arange(n, device=device)
        current_dims = self.box_dims[env_idx, idx]  # (N, 3)
        
        # 6. Proprioception (placeholder - would come from robot)
        proprio = torch.zeros(n, self.cfg.robot_state_dim, device=device)
        
        # 7. Concatenate all observations
        obs = torch.cat([heightmap_flat, buffer_flat, current_dims, proprio], dim=-1)
        
        # Shape/device assertion (fast, only runs in debug or once per step)
        assert obs.shape == (n, self.cfg.num_observations), \
            f"Obs shape {obs.shape} != expected ({n}, {self.cfg.num_observations})"
        assert obs.device == device, \
            f"Obs device {obs.device} != expected {device}"
        
        return {"policy": obs, "critic": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """
        Compute rewards (pure PyTorch, JIT-compatible).
        Also computes task KPIs logged via self.extras for TensorBoard.
        
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
            
            target_pos = self.last_target_pos[valid_envs]
            dist = torch.norm(current_pos[:, :2] - target_pos[:, :2], dim=-1)
            fell = current_pos[:, 2] < 0.05
            unstable = dist > 0.10
            
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
        # Compute and log task KPIs (mean scalars for TensorBoard)
        # =====================================================================
        place_attempts = (self.active_place_mask & ~self.valid_retrieve).float().sum()
        retrieve_attempts = self.retrieve_mask.float().sum()
        store_attempts = self.store_mask.float().sum()
        
        # Initialize extras dict if not present
        if not hasattr(self, 'extras'):
            self.extras = {}
        
        # KPI #1: Place success rate (among PLACE actions)
        place_success_rate = place_success.float().sum() / (place_attempts + 1e-8)
        self.extras["metrics/place_success_rate"] = place_success_rate
        
        # KPI #2: Place failure rate (among PLACE actions)
        place_failure_rate = place_failure.float().sum() / (place_attempts + 1e-8)
        self.extras["metrics/place_failure_rate"] = place_failure_rate
        
        # KPI #3: Retrieve success rate (among RETRIEVE actions)
        retrieve_success_rate = retrieve_success.float().sum() / (retrieve_attempts + 1e-8)
        self.extras["metrics/retrieve_success_rate"] = retrieve_success_rate
        
        # KPI #4: Store accept rate (fraction of STORE that were valid)
        # valid_store was computed in _handle_buffer_actions; approximated here by
        # checking how many store envs actually wrote to buffer
        # We track using store_mask - we need the accepted count from _handle_buffer_actions
        # For now, use a simple metric: stores that had box_idx > 0 and empty slot
        has_box_to_store = (self.box_idx > 0).float()
        store_accept_approx = (self.store_mask.float() * has_box_to_store).sum() / (store_attempts + 1e-8)
        self.extras["metrics/store_accept_rate"] = store_accept_approx
        
        # KPI #5: Buffer occupancy (mean fraction of slots occupied)
        buffer_occupancy = self.buffer_has_box.float().mean()
        self.extras["metrics/buffer_occupancy"] = buffer_occupancy
        
        return rewards
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute termination and truncation flags.
        
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
            # Optimization: only read box state for valid envs
            valid_envs = valid_eval.nonzero(as_tuple=False).flatten()
            eval_box_idx = self.last_moved_box_id[valid_envs]
            global_idx = valid_envs * self.cfg.max_boxes + eval_box_idx
            current_pos = self.scene["boxes"].data.root_pos_w[global_idx]
            
            target_pos = self.last_target_pos[valid_envs]
            dist = torch.norm(current_pos[:, :2] - target_pos[:, :2], dim=-1)
            fell = current_pos[:, 2] < 0.05
            unstable = dist > 0.10
            
            # Compute failure for valid envs, expand to full tensor
            failure_valid = fell | unstable
            active_place_valid = self.active_place_mask[valid_envs]
            
            # Only terminate if valid placement occurred and it failed
            terminated[valid_envs] = active_place_valid & failure_valid
        
        # Check if all boxes used (box_idx already incremented on Place, so >= means done)
        terminated = terminated | (self.box_idx >= self.cfg.max_boxes)
        
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
        self.last_target_pos[env_ids] = 0.0
        
        # Randomize box dimensions for new episode
        base_dims = torch.tensor([0.4, 0.3, 0.2], device=device)
        rand_offset = torch.rand(len(env_ids), self.cfg.max_boxes, 3, device=device) * 0.2 - 0.1
        self.box_dims[env_ids] = base_dims + rand_offset
        
        # Move all boxes off-map initially (they'll be placed when actions are taken)
        # This ensures inactive boxes never pollute the heightmap
        if "boxes" in self.scene.keys():
            n_total = len(env_ids) * self.cfg.max_boxes
            inactive_pos = self._inactive_box_pos.expand(n_total, 3)
            inactive_quat = torch.zeros(n_total, 4, device=device)
            inactive_quat[:, 0] = 1.0  # Identity quaternion (w,x,y,z)
            
            # Compute global indices for all boxes in reset envs (vectorized, GPU-safe)
            env_ids_long = env_ids.long()
            base = env_ids_long * self.cfg.max_boxes  # (num_reset_envs,)
            offsets = torch.arange(self.cfg.max_boxes, device=device)  # (max_boxes,)
            global_indices = (base[:, None] + offsets[None, :]).reshape(-1)  # (num_reset_envs * max_boxes,)
            
            self.scene["boxes"].write_root_pose_to_sim(
                inactive_pos, inactive_quat, indices=global_indices
            )
    
    def _apply_action(self, action: torch.Tensor):
        """
        Apply MultiDiscrete action.
        
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
        
        # Compute target position (5cm grid)
        step = 0.05
        target_x = grid_x.float() * step - 0.4 + step / 2
        target_y = grid_y.float() * step - 0.6 + step / 2
        target_z = torch.full((n,), 1.5, device=device)  # Drop height
        
        # Operation masks
        # PLACE (0) and RETRIEVE (2) both result in a box being on the pallet after physics
        # but ONLY PLACE should move box_idx here; RETRIEVE moves its physical box in _handle_buffer_actions
        self.active_place_mask = (op_type == 0) | (op_type == 2)
        self.store_mask = (op_type == 1)
        self.retrieve_mask = (op_type == 2)
        
        # Mask for PLACE-only envs (for writing box_idx pose)
        place_only_mask = (op_type == 0)
        
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
        
        # Apply to simulation: ONLY move box_idx for PLACE actions
        # STORE and RETRIEVE handle their own box movements in _handle_buffer_actions
        if "boxes" in self.scene.keys() and place_only_mask.any():
            place_envs = place_only_mask.nonzero(as_tuple=False).flatten()
            global_idx = place_envs * self.cfg.max_boxes + self.box_idx[place_envs]
            
            self.scene["boxes"].write_root_pose_to_sim(
                target_pos[place_envs], quat[place_envs], indices=global_idx
            )
            
            vel = torch.zeros(len(place_envs), 6, device=device)
            self.scene["boxes"].write_root_velocity_to_sim(vel, indices=global_idx)
        
        # Buffer logic for store/retrieve
        self._handle_buffer_actions(action)
    
    def _handle_buffer_actions(self, action: torch.Tensor):
        """
        Handle store and retrieve buffer operations with physical box tracking.
        
        Physical buffer semantics:
        - STORE parks an existing physical box in a holding area and records its ID
        - RETRIEVE moves that same parked physical box back onto the pallet
        - The buffer does NOT create new boxes
        
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
        
        if valid_store.any():
            store_envs = valid_store.nonzero(as_tuple=False).flatten()
            store_slots = slot_idx[store_envs]
            
            # The physical box being stored is the last placed: box_idx - 1
            stored_physical_id = (self.box_idx[store_envs] - 1).clamp(0, self.cfg.max_boxes - 1)
            dims = self.box_dims[store_envs, stored_physical_id]
            
            # Update buffer_state (dims, active flag, age)
            self.buffer_state[store_envs, store_slots, :3] = dims
            self.buffer_state[store_envs, store_slots, 3] = 1.0  # Active flag
            self.buffer_state[store_envs, store_slots, 4] = 0.0  # Reset age
            
            # Track physical box identity in slot
            self.buffer_has_box[store_envs, store_slots] = True
            self.buffer_box_id[store_envs, store_slots] = stored_physical_id
            
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
        has_box_in_slot = self.buffer_has_box[env_idx, slot_idx]
        self.valid_retrieve = self.retrieve_mask & has_box_in_slot
        
        if self.valid_retrieve.any():
            retr_envs = self.valid_retrieve.nonzero(as_tuple=False).flatten()
            retr_slots = slot_idx[retr_envs]
            
            # Get the physical box ID from the buffer slot
            retrieved_physical_id = self.buffer_box_id[retr_envs, retr_slots]
            
            # Move THAT physical box to the target position
            # Note: target position was already computed in _apply_action
            if "boxes" in self.scene.keys():
                global_retr_idx = retr_envs * self.cfg.max_boxes + retrieved_physical_id
                
                # Build target pose for retrieved boxes
                step = 0.05
                grid_x = action[retr_envs, 2]
                grid_y = action[retr_envs, 3]
                rot_idx = action[retr_envs, 4]
                
                target_x = grid_x.float() * step - 0.4 + step / 2
                target_y = grid_y.float() * step - 0.6 + step / 2
                target_z = torch.full((len(retr_envs),), 1.5, device=device)
                target_pos = torch.stack([target_x, target_y, target_z], dim=-1)
                
                quat = torch.zeros(len(retr_envs), 4, device=device)
                quat[:, 0] = 1.0
                rot_mask = (rot_idx == 1)
                quat[rot_mask, 0] = 0.7071068
                quat[rot_mask, 3] = 0.7071068
                
                self.scene["boxes"].write_root_pose_to_sim(
                    target_pos, quat, indices=global_retr_idx
                )
                
                vel = torch.zeros(len(retr_envs), 6, device=device)
                self.scene["boxes"].write_root_velocity_to_sim(vel, indices=global_retr_idx)
            
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
        place_mask = (op_type == 0)
        
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

        The current implementation marks all actions as valid (all True),
        but the interface is provided so that task-specific constraints
        (e.g. collision-based invalid placements) can be injected without
        changing the policy class.
        
        This mask is compatible with PalletizerActorCritic.act() and evaluate()
        methods, which expect (N, total_logits) bool tensors.
        """
        n = self.num_envs
        total_logits = sum(self.cfg.action_dims)
        # Ensure correct dtype, shape, and device for policy compatibility
        mask = torch.ones(n, total_logits, dtype=torch.bool, device=self._device)
        assert mask.device == self._device, "Action mask device mismatch"
        return mask
