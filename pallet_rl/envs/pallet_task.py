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
    decimation: int = 10
    
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
        
        # Buffer state: (N, buffer_slots, buffer_features)
        self.buffer_state = torch.zeros(
            n, self.cfg.buffer_slots, self.cfg.buffer_features, device=device
        )
        
        # Current box index per env
        self.box_idx = torch.zeros(n, dtype=torch.long, device=device)
        
        # Masks for reward computation
        self.active_place_mask = torch.zeros(n, dtype=torch.bool, device=device)
        self.store_mask = torch.zeros(n, dtype=torch.bool, device=device)
        self.retrieve_mask = torch.zeros(n, dtype=torch.bool, device=device)
        self.valid_retrieve = torch.zeros(n, dtype=torch.bool, device=device)
        
        # Target position for stability check
        self.last_target_pos = torch.zeros(n, 3, device=device)
    
    def _setup_scene(self):
        """
        Configure scene objects using Isaac Lab declarative API.
        
        Scene contains:
        - Pallet (static rigid body)
        - Boxes (rigid object collection)
        - Robot (articulation) - optional, can be added in subclass
        """
        # Note: In Isaac Lab 4.0+, scene config is typically declarative
        # through InteractiveSceneCfg. This method can add dynamic objects.
        
        # For now, we assume boxes are configured in scene and accessed via
        # self.scene["boxes"]. The actual spawning is done by Isaac Lab.
        pass
    
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
            all_rot = self.scene["boxes"].data.root_quat_w  # (N*max_boxes, 4)
            
            box_pos = all_pos.view(n, self.cfg.max_boxes, 3)
            box_rot = all_rot.view(n, self.cfg.max_boxes, 4)
        else:
            # Fallback for testing
            box_pos = torch.zeros(n, self.cfg.max_boxes, 3, device=device)
            box_rot = torch.zeros(n, self.cfg.max_boxes, 4, device=device)
            box_rot[:, :, 0] = 1.0  # Identity quat w=1
        
        # Pallet positions (centered at origin per env)
        pallet_pos = torch.zeros(n, 3, device=device)
        
        # 2. Generate heightmap (GPU-only via Warp)
        heightmap = self.heightmap_gen.forward(
            box_pos.reshape(-1, 3),
            box_rot.reshape(-1, 4),
            self.box_dims.reshape(-1, 3),
            pallet_pos
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
        
        # Shape assertion
        assert obs.shape == (n, self.cfg.num_observations), \
            f"Obs shape {obs.shape} != expected ({n}, {self.cfg.num_observations})"
        
        return {"policy": obs, "critic": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """
        Compute rewards (pure PyTorch, JIT-compatible).
        
        Returns:
            rewards: (N,) tensor
        """
        n = self.num_envs
        device = self._device
        
        rewards = torch.zeros(n, device=device)
        
        # Penalty for storing (using buffer)
        rewards = rewards - 0.1 * self.store_mask.float()
        
        # Bonus for successful retrieve
        rewards = rewards + 2.0 * self.valid_retrieve.float()
        
        # Buffer age penalty
        ages = self.buffer_state[:, :, 4].sum(dim=1)  # Sum of ages
        rewards = rewards - 0.01 * ages
        
        # Success/failure for placement
        if "boxes" in self.scene.keys():
            global_idx = torch.arange(n, device=device) * self.cfg.max_boxes + self.box_idx
            global_idx = global_idx.clamp(0, n * self.cfg.max_boxes - 1)
            current_pos = self.scene["boxes"].data.root_pos_w[global_idx]
            
            dist = torch.norm(current_pos[:, :2] - self.last_target_pos[:, :2], dim=-1)
            fell = current_pos[:, 2] < 0.05
            unstable = dist > 0.10
            
            failure = self.active_place_mask & (fell | unstable)
            success = self.active_place_mask & ~failure
            
            rewards = rewards - 10.0 * failure.float()
            rewards = rewards + 1.0 * success.float()
            
            # Volume bonus for successful placement
            idx = self.box_idx.clamp(0, self.cfg.max_boxes - 1)
            dims = self.box_dims[torch.arange(n, device=device), idx]
            vol = dims[:, 0] * dims[:, 1] * dims[:, 2]
            rewards = rewards + vol * success.float()
        
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
        if "boxes" in self.scene.keys():
            global_idx = torch.arange(n, device=device) * self.cfg.max_boxes + self.box_idx
            global_idx = global_idx.clamp(0, n * self.cfg.max_boxes - 1)
            current_pos = self.scene["boxes"].data.root_pos_w[global_idx]
            
            dist = torch.norm(current_pos[:, :2] - self.last_target_pos[:, :2], dim=-1)
            fell = current_pos[:, 2] < 0.05
            unstable = dist > 0.10
            
            terminated = self.active_place_mask & (fell | unstable)
        
        # Check if all boxes used
        terminated = terminated | (self.box_idx >= self.cfg.max_boxes)
        
        # Truncation from time limit handled by base class
        truncated = self.episode_length_buf >= self.max_episode_length
        
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
        self.box_idx[env_ids] = 0
        self.active_place_mask[env_ids] = False
        self.store_mask[env_ids] = False
        self.retrieve_mask[env_ids] = False
        self.valid_retrieve[env_ids] = False
        self.last_target_pos[env_ids] = 0.0
        
        # Randomize box dimensions for new episode
        base_dims = torch.tensor([0.4, 0.3, 0.2], device=device)
        rand_offset = torch.rand(len(env_ids), self.cfg.max_boxes, 3, device=device) * 0.2 - 0.1
        self.box_dims[env_ids] = base_dims + rand_offset
    
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
        self.active_place_mask = (op_type == 0) | (op_type == 2)
        self.store_mask = (op_type == 1)
        self.retrieve_mask = (op_type == 2)
        
        # Build target position tensor
        target_pos = torch.stack([target_x, target_y, target_z], dim=-1)
        holding_pos = torch.tensor([100.0, 100.0, 100.0], device=device)
        
        final_pos = torch.where(
            self.active_place_mask.unsqueeze(-1),
            target_pos,
            holding_pos.expand(n, 3)
        )
        
        # Rotation quaternion
        quat = torch.zeros(n, 4, device=device)
        quat[:, 0] = 1.0  # Identity (w, x, y, z)
        rot_mask = (rot_idx == 1)
        quat[rot_mask, 0] = 0.7071068  # 90° Z rotation
        quat[rot_mask, 3] = 0.7071068
        
        # Save target for stability check
        self.last_target_pos[:, 0] = target_x
        self.last_target_pos[:, 1] = target_y
        self.last_target_pos[:, 2] = 0.0
        
        # Apply to simulation (if boxes in scene)
        if "boxes" in self.scene.keys():
            env_idx = torch.arange(n, device=device)
            global_idx = env_idx * self.cfg.max_boxes + self.box_idx
            
            self.scene["boxes"].write_root_pose_to_sim(
                final_pos, quat, indices=global_idx
            )
            
            vel = torch.zeros(n, 6, device=device)
            self.scene["boxes"].write_root_velocity_to_sim(vel, indices=global_idx)
        
        # Buffer logic for store/retrieve
        self._handle_buffer_actions(action)
    
    def _handle_buffer_actions(self, action: torch.Tensor):
        """Handle store and retrieve buffer operations."""
        device = self._device
        n = self.num_envs
        env_idx = torch.arange(n, device=device)
        
        slot_idx = action[:, 1].long()
        
        # Store: save current box dims to buffer
        if self.store_mask.any():
            store_envs = self.store_mask.nonzero(as_tuple=False).flatten()
            store_slots = slot_idx[store_envs]
            dims = self.box_dims[store_envs, self.box_idx[store_envs]]
            
            self.buffer_state[store_envs, store_slots, :3] = dims
            self.buffer_state[store_envs, store_slots, 3] = 1.0  # ID = active
            self.buffer_state[store_envs, store_slots, 4] = 0.0  # Reset age
        
        # Retrieve: get box dims from buffer
        has_data = self.buffer_state[env_idx, slot_idx, 3] > 0.5
        self.valid_retrieve = self.retrieve_mask & has_data
        
        if self.valid_retrieve.any():
            retr_envs = self.valid_retrieve.nonzero(as_tuple=False).flatten()
            retr_slots = slot_idx[retr_envs]
            
            # Copy dims from buffer to current box
            self.box_dims[retr_envs, self.box_idx[retr_envs]] = \
                self.buffer_state[retr_envs, retr_slots, :3]
            
            # Clear buffer slot
            self.buffer_state[retr_envs, retr_slots] = 0.0
            
            # This counts as a placement too
            self.active_place_mask = self.active_place_mask | self.valid_retrieve
        
        # Age all buffer slots
        self.buffer_state[:, :, 4] += 1.0
    
    def step(self, action: torch.Tensor):
        """
        Execute one environment step.
        
        Args:
            action: (N, 5) MultiDiscrete action tensor
            
        Returns:
            obs, reward, terminated, truncated, info
        """
        # Apply action
        self._apply_action(action)
        
        # Step physics (settling loop)
        for _ in range(50):
            self.sim.step(render=self._render)
        
        # Advance box index
        self.box_idx += 1
        
        # Get observations
        obs = self._get_observations()
        
        # Get rewards
        rewards = self._get_rewards()
        
        # Get termination signals
        terminated, truncated = self._get_dones()
        
        # Build info dict
        info = {"time_outs": truncated}
        
        # Handle resets
        reset_mask = terminated | truncated
        if reset_mask.any():
            reset_ids = reset_mask.nonzero(as_tuple=False).flatten()
            self._reset_idx(reset_ids)
        
        return obs, rewards, terminated, truncated, info
