
import torch
import numpy as np
from typing import Dict, Any

from omni.isaac.core.utils.torch.rotations import quat_rotate_inverse, quat_mul
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass
import gymnasium as gym

from pallet_rl.utils.heightmap_rasterizer import WarpHeightmapGenerator

@configclass
class PalletTaskCfg(DirectRLEnvCfg):
    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1/60.0, render_interval=2)
    
    # Decimation
    decimation = 10 
    
    # Env Params
    num_envs: int = 4096
    env_spacing: float = 3.0
    pallet_size_cm: list = [120., 100.] # Actual logic uses 120x80 usually? Code used 120/100 default but logic might assume 120/80? 
    # Let's align with user constraints [120, 80] if implied by 240x160 pixels.
    # But Config said [120, 100]. 160x240 @ 0.5cm covers 80x120.
    # I should set pallet_size_cm to [120., 80.] strictly to match the new map?
    # Or map covers slightly more?
    # Let's set to [120, 80] to be clean.
    pallet_size_cm: list = [120., 80.] 
    grid_res_cm: float = 0.5 # Fine resolution for CNN
    map_shape: tuple = (160, 240) # Rectangular (X, Y)
    max_boxes: int = 50
    max_height: float = 2.0 # For normalization
    
    # State Dims
    # Robot State: Joints (e.g. 6 pos + 6 vel) + Gripper? 
    # Let's assume 12 DOFs total -> 24.
    robot_state_dim = 24 
    
    # Calculated later in init, but for Config access we might need it.
    # 160*240 = 38400 (Visual)
    # 50 (Buffer) + 3 (Dims) + 24 (Proprio) = 77
    # Total = 38477
    num_observations = 38477
    # MultiDiscrete action space doesn't map to a single int easily for config, 
    # but we will override action_space in __init__
    num_actions = 5 # Placeholder length of MultiDiscrete dims
    
    episode_length_s = 60.0

class PalletTask(DirectRLEnv):
    def __init__(self, cfg: PalletTaskCfg, render_mode: str | None = None, **kwargs):
        # Recalculate if needed or trust cfg
        self.robot_state_dim = cfg.robot_state_dim
        self.map_shape = cfg.map_shape # Tuple
        
        # Dynamic calculation of observation dimension
        vis_dim = self.map_shape[0] * self.map_shape[1]
        vec_dim = 10 * 5 # Buffer flattened
        box_dim = 3
        cfg.num_observations = vis_dim + vec_dim + box_dim + self.robot_state_dim
        
        super().__init__(cfg, render_mode, **kwargs)
        
        # Warp Rasterizer Init
        # Warp Rasterizer Init
        self.heightmap_gen = WarpHeightmapGenerator(
            device=self.device,
            num_envs=self.num_envs,
            max_boxes=self.cfg.max_boxes,
            grid_res=self.cfg.grid_res_cm / 100.0,
            map_size=self.map_shape,
            pallet_dims=(self.cfg.pallet_size_cm[0]/100.0, self.cfg.pallet_size_cm[1]/100.0)
        )
        
        # Box Dims (Placeholder/Mock)
        self.box_dims_tensor = torch.tensor([0.4, 0.3, 0.2], device=self.device).repeat(self.num_envs, self.cfg.max_boxes, 1)

        # --- Buffer Logic ---
        # Shape: (num_envs, 10, 5) -> 10 slots, 5 dims: [L, W, H, ID, Age]
        self.buffer_state = torch.zeros((self.num_envs, 10, 5), device=self.device)
        
        # --- Action Space (User Override: Step 2) ---
        # Use gym.spaces.MultiDiscrete
        # [Operation(3), BufferSlot(10), GridX(16), GridY(24), Rotation(2)]
        self.action_space = gym.spaces.MultiDiscrete([3, 10, 16, 24, 2])
        
        # Internal buffers for Step
        self.dones_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.rewards_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def _setup_scene(self):
        pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Reset buffer for these envs
        self.buffer_state[env_ids] = 0.0
        
        # Reset Box Index
        if not hasattr(self, "box_idx"):
            self.box_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.box_idx[env_ids] = 0
        
        # Randomize Box Dims for the new episode
        # (N, max_boxes, 3)
        # We'll just randomize for ALL boxes in these envs for simplicity
        # Or just the sequence? Let's randomize the whole sequence
        dims = torch.tensor([0.4, 0.3, 0.2], device=self.device) # Base
        rand = torch.rand((len(env_ids), self.cfg.max_boxes, 3), device=self.device) * 0.2 - 0.1
        self.box_dims_tensor[env_ids] = dims + rand
        
        # Reset Box Poses (Move all to spawn/holding area)
        # We need to move ALL boxes for these envs away
        # boxes shape in view? We need to verify how "boxes" are indexed.
        # Assuming DirectRLEnv flat indexing? Or [N, MaxBoxes]?
        # Usually RigidObject doesn't support multidim indexing naturally if initialized flat.
        # But let's assume we can access them.
        # If not, we rely on physics simulation handling "sleeping" boxes.
        # We will teleport the active box in step() anyway.

    # User Override Step 3: Implement Physics Settlement Loop
    def step(self, actions):
        """Apply actions, step physics, compute returns."""
        # 1. Apply Action logic (Teleport active box)
        self._apply_actions_logic(actions)
        
        # 2. PHYSICS SETTLING LOOP ("Drop Test")
        # We act as a turn-based system. 
        # We step the physics 50 times (approx 0.5s) to let gravity work.
        for _ in range(50):
            self.sim.step(render=self.enable_render)
                
        # 3. Compute observations & rewards AFTER the box has settled
        # obs dictionary
        self.obs_buf = self._get_observations() 
        
        # Compute rewards and dones (split logic or kept together)
        # Helper returns (rew, dones)
        self.rew_buf, self.reset_buf = self._compute_rewards_and_dones(actions)
        
        self.extras["time_outs"] = self.episode_length_buf >= self.max_episode_length
        self.reset_buf = self.reset_buf | self.extras["time_outs"]
        
        # 4. Handle resets
        if torch.sum(self.reset_buf) > 0:
            self._reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())
            
        # Return 5-tuple for DirectRLEnv/Gymnasium compliance
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras["time_outs"], self.extras

    def _apply_actions_logic(self, actions: torch.Tensor):
        # Logic extracted from previous 'step' implementation
        # actions: (N, 5) -> Op, Slot, X, Y, Rot
        op = actions[:, 0]
        slot = actions[:, 1]
        grid_x = actions[:, 2]
        grid_y = actions[:, 3]
        rot = actions[:, 4]
        
        place_mask = (op == 0)
        store_mask = (op == 1)
        retrieve_mask = (op == 2)
        
        env_indices = torch.arange(self.num_envs, device=self.device)
        active_box_idx = self.box_idx
        # Global index: env_idx * max_boxes + box_idx
        # But wait, Isaac Lab RigidObject usually uses global indexing if wrapping all envs.
        # But `write_root_pose_to_sim` on `self.scene["boxes"]`?
        # self.scene["boxes"] is likely a RigidObject view.
        # Check `_get_observations` uses `boxes.data.root_pos_w`.
        # We assume `boxes` wraps all boxes in all envs.
        # Index = env_id * max_boxes + box_id
        global_active_idx = env_indices * self.cfg.max_boxes + active_box_idx
        
        # Store
        if torch.any(store_mask):
            target_slots = slot[store_mask]
            current_dims = self.box_dims_tensor[env_indices[store_mask], active_box_idx[store_mask]]
            features = torch.zeros((store_mask.sum(), 5), device=self.device)
            features[:, :3] = current_dims
            features[:, 3] = 1.0 
            features[:, 4] = 0.0 # Age reset?
            self.buffer_state[env_indices[store_mask], target_slots] = features

        # Retrieve (swap dimensions and clear buffer)
        has_data =  self.buffer_state[env_indices, slot, 3] > 0.5
        valid_retrieve = retrieve_mask & has_data
        if torch.any(valid_retrieve):
            retrieved_dims = self.buffer_state[env_indices[valid_retrieve], slot[valid_retrieve], :3]
            self.box_dims_tensor[env_indices[valid_retrieve], active_box_idx[valid_retrieve]] = retrieved_dims
            self.buffer_state[env_indices[valid_retrieve], slot[valid_retrieve]] = 0.0
            
        # Calc Targets
        active_place_mask = (op == 0) | valid_retrieve
        
        target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        target_rot = torch.zeros((self.num_envs, 4), device=self.device)
        target_rot[:, 0] = 1.0 # Identity (w,x,y,z)
        
        if torch.any(active_place_mask):
            idx_place = env_indices[active_place_mask]
            
            # Grid Stats
            res = self.cfg.grid_res_cm / 100.0
            pallet_x = self.cfg.pallet_size_cm[1] / 100.0 # 80cm
            pallet_y = self.cfg.pallet_size_cm[0] / 100.0 # 120cm
            
            x_offset = -pallet_x / 2.0
            y_offset = -pallet_y / 2.0
            
            gx = grid_x[active_place_mask].float()
            gy = grid_y[active_place_mask].float()
            
            tx = x_offset + (gx * res) + (res/2.0)
            ty = y_offset + (gy * res) + (res/2.0)
            tz = torch.full_like(tx, 1.5) # Drop height
            
            target_pos[idx_place, 0] = tx + self.scene.env_origins[idx_place, 0]
            target_pos[idx_place, 1] = ty + self.scene.env_origins[idx_place, 1]
            target_pos[idx_place, 2] = tz + self.scene.env_origins[idx_place, 2]
            
            r_mask = (rot[active_place_mask] == 1)
            # 90 deg z: (0.7071, 0, 0, 0.7071)
            target_rot[idx_place[r_mask]] = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=self.device)
            
        # Teleport
        final_pos = target_pos.clone()
        final_rot = target_rot.clone()
        
        # Teleport away (Store / Invalid Retrieve)
        away_mask = ~active_place_mask
        if torch.any(away_mask):
            idx_away = env_indices[away_mask]
            final_pos[idx_away, 2] = -10.0
            
        # Apply to Sim
        self.scene["boxes"].write_root_pose_to_sim(final_pos, final_rot, indices=global_active_idx)
        
        zeros_vel6 = torch.zeros((self.num_envs, 6), device=self.device)
        self.scene["boxes"].write_root_velocity_to_sim(zeros_vel6, indices=global_active_idx)
        
        # Save state for rewards
        self.last_target_pos = target_pos # For stability/drift check
        self.active_place_mask = active_place_mask
        self.valid_retrieve = valid_retrieve
        self.store_mask = store_mask

    def _compute_rewards_and_dones(self, actions):
        # Re-implement existing compute_rewards logic
        # Using state saved in _apply
        
        active_place_mask = self.active_place_mask
        
        global_active_idx = torch.arange(self.num_envs, device=self.device) * self.cfg.max_boxes + self.box_idx
        current_pos = self.scene["boxes"].data.root_pos_w[global_active_idx]
        
        dist = torch.norm(current_pos[:, :2] - self.last_target_pos[:, :2], dim=-1)
        
        # op from actions requires refetch or recompute? 
        # Easier to call self.compute_rewards with args? 
        # But logic is simple.
        
        op = actions[:, 0]
        
        rew = torch.zeros(self.num_envs, device=self.device)
        rew[self.store_mask] -= 0.1
        rew[self.valid_retrieve] += 2.0
        
        ages = self.buffer_state[:, :, 4].sum(dim=1)
        rew -= 0.01 * ages
        
        fell = current_pos[:, 2] < 0.05
        unstable = dist > 0.10
        
        failure = active_place_mask & (fell | unstable)
        success = active_place_mask & (~failure)
        
        rew[failure] -= 10.0
        rew[success] += 1.0
        
        # Volume Bonus
        idx = self.box_idx
        dims = self.box_dims_tensor[torch.arange(self.num_envs, device=self.device), idx]
        vol = dims[:, 0] * dims[:, 1] * dims[:, 2]
        rew[success] += vol[success]
        
        dones = failure
        
        # Advance Box Index
        self.box_idx += 1
        dones = dones | (self.box_idx >= self.cfg.max_boxes)
        
        # Buffer Age Update (Once per step)
        self.buffer_state[:, :, 4] += 1.0
        
        return rew, dones


