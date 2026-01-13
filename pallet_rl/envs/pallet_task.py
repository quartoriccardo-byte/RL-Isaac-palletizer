
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
    @property
    def num_observations(self):
        # 160x240 (Heightmap) + 53 (Buffer/State Vector)
        return 38453
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
        """
        Scene setup for PalletTask.
        
        ARCHITECTURE NOTE:
        Isaac Lab DirectRLEnv uses declarative scene configuration via
        InteractiveSceneCfg. The 'boxes' RigidObjectCollection should be
        pre-configured in the scene config, not created here dynamically.
        
        This method is intentionally minimal as the scene objects are:
        1. Defined in PalletTaskCfg.scene (InteractiveSceneCfg)
        2. Created by DirectRLEnv.__init__() before calling this method
        
        If you need to add objects dynamically, do so here using:
            self.scene.add_object(...)
        """
        # Scene is configured declaratively - no dynamic setup needed
        # The 'boxes' rigid object collection must exist in self.scene
        pass
    
    def _get_observations(self) -> dict:
        """
        Constructs observation dict required by RSL-RL wrapper.
        
        Observation structure (flattened):
            [Heightmap (38400) | Buffer (50) | BoxDims (3) | Proprio (24)]
            Total: 38477 (matches cfg.num_observations)
        
        Returns:
            dict: {'policy': Tensor[N, obs_dim], 'critic': Tensor[N, obs_dim]}
        """
        # 1. Generate heightmap from current box positions
        # Access box poses from scene - shape depends on how boxes are registered
        # Assuming boxes are registered as RigidObjectCollection with shape (N*max_boxes, ...)
        if "boxes" in self.scene.keys():
            all_box_pos = self.scene["boxes"].data.root_pos_w  # (N*max_boxes, 3)
            all_box_rot = self.scene["boxes"].data.root_quat_w  # (N*max_boxes, 4)
            
            # Reshape to (N, max_boxes, dim)
            box_pos = all_box_pos.view(self.num_envs, self.cfg.max_boxes, 3)
            box_rot = all_box_rot.view(self.num_envs, self.cfg.max_boxes, 4)
        else:
            # Fallback for testing without scene
            box_pos = torch.zeros(self.num_envs, self.cfg.max_boxes, 3, device=self.device)
            box_rot = torch.zeros(self.num_envs, self.cfg.max_boxes, 4, device=self.device)
            box_rot[:, :, 0] = 1.0  # Identity quaternion w=1
        
        # Pallet positions (centered at origin for each env)
        pallet_pos = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Generate heightmap using Warp rasterizer
        heightmap = self.heightmap_gen.forward(
            box_pos.view(-1, 3),  # Flatten for Warp
            box_rot.view(-1, 4),
            self.box_dims_tensor.view(-1, 3),
            pallet_pos
        )  # Returns (N, H, W)
        
        # 2. Normalize heightmap to [0, 1]
        heightmap_norm = heightmap / self.cfg.max_height
        heightmap_flat = heightmap_norm.view(self.num_envs, -1)  # (N, 38400)
        
        # 3. Buffer state (N, 10, 5) -> (N, 50)
        buffer_flat = self.buffer_state.view(self.num_envs, -1)
        
        # 4. Current box dimensions (N, 3)
        idx = self.box_idx.clamp(0, self.cfg.max_boxes - 1)
        env_indices = torch.arange(self.num_envs, device=self.device)
        current_dims = self.box_dims_tensor[env_indices, idx]
        
        # 5. Proprioceptive state (placeholder - robot state would come from scene)
        proprio = torch.zeros(self.num_envs, self.robot_state_dim, device=self.device)
        
        # 6. Concatenate all components
        obs = torch.cat([heightmap_flat, buffer_flat, current_dims, proprio], dim=-1)
        
        # Shape assertion for early failure detection
        expected_dim = self.map_shape[0] * self.map_shape[1] + 50 + 3 + self.robot_state_dim
        assert obs.shape[-1] == expected_dim, \
            f"Observation shape mismatch: got {obs.shape[-1]}, expected {expected_dim}"
        
        return {"policy": obs, "critic": obs}

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
        # actions shape: [num_envs, 5] -> [Op, Slot, GridX, GridY, Rot]
        
        # 1. Parse Discrete Actions
        op_type = actions[:, 0]
        # slot_idx = actions[:, 1] # Used for buffer logic later
        grid_x = actions[:, 2]
        grid_y = actions[:, 3]
        rot_idx = actions[:, 4]

        # 2. Calculate Physical Target (Grid Resolution = 5cm)
        # Assuming Pallet center is at (0,0) and size is 80x120cm
        # We map 0..15 (X) to -0.4..0.4 and 0..23 (Y) to -0.6..0.6
        x_step = 0.05
        y_step = 0.05
        
        # Calculate offsets relative to pallet corner
        target_x = (grid_x * x_step) - 0.4 + (x_step / 2)
        target_y = (grid_y * y_step) - 0.6 + (y_step / 2)
        target_z = 1.5  # Spawn high above pallet
        
        # 3. Apply Teleportation
        # We only move the box if Op == 0 (Place) or 2 (Retrieve)
        # If Op == 1 (Store), we move it away to a holding area
        
        should_place = (op_type == 0) | (op_type == 2)
        
        # Create position tensor
        pos = torch.zeros((self.num_envs, 3), device=self.device)
        pos[:, 0] = target_x
        pos[:, 1] = target_y
        pos[:, 2] = target_z
        
        # Handle "Storage" action (Hide box far away)
        # If op_type == 1, move to (100, 100, 100)
        holding_pos = torch.tensor([100.0, 100.0, 100.0], device=self.device)
        final_pos = torch.where(should_place.unsqueeze(-1), pos, holding_pos)
        
        # Orientation (Rotate 90 deg around Z if rot_idx == 1)
        # Identity quaternion is [1, 0, 0, 0] (w, x, y, z)
        # 90 deg Z rotation quaternion is [0.707, 0, 0, 0.707]
        quat = torch.zeros((self.num_envs, 4), device=self.device)
        quat[:, 0] = 1.0 # Default w
        
        # Apply rotation mask
        rotate_mask = (rot_idx == 1)
        quat[rotate_mask, 0] = 0.7071068
        quat[rotate_mask, 3] = 0.7071068
        
        # Identify Masks for Reward/Logic (Needed by compute_rewards)
        self.active_place_mask = (op_type == 0)
        self.store_mask = (op_type == 1)
        self.retrieve_mask = (op_type == 2)
        
        # Retrieve buffer logic needs to update box data if Retrieve op
        # Check buffer validity
        has_data =  self.buffer_state[torch.arange(self.num_envs, device=self.device), actions[:, 1].long(), 3] > 0.5
        self.valid_retrieve = self.retrieve_mask & has_data
        
        # If Retrieve and Valid, update box dimensions from Buffer
        # AND Clear buffer
        if torch.any(self.valid_retrieve):
             env_ids = self.valid_retrieve.nonzero(as_tuple=False).flatten()
             slots = actions[env_ids, 1].long()
             # retrieved_dims = self.buffer_state[env_ids, slots, :3]
             # Update logic dims
             self.box_dims_tensor[env_ids, self.box_idx[env_ids]] = self.buffer_state[env_ids, slots, :3]
             # Clear
             self.buffer_state[env_ids, slots] = 0.0
             
             # Also update active_place_mask for rewards to include successful retrieves
             self.active_place_mask = self.active_place_mask | self.valid_retrieve

        # Store Logic
        if torch.any(self.store_mask):
             env_ids = self.store_mask.nonzero(as_tuple=False).flatten()
             slots = actions[env_ids, 1].long()
             # Dims
             dims = self.box_dims_tensor[env_ids, self.box_idx[env_ids]]
             # Write to buffer
             # [L, W, H, ID, Age]
             self.buffer_state[env_ids, slots, :3] = dims
             self.buffer_state[env_ids, slots, 3] = 1.0
             self.buffer_state[env_ids, slots, 4] = 0.0
        
        # Apply to Sim
        # Note: self.active_box is not defined in previous code. 
        # Previous code used self.scene["boxes"]
        # We must use self.scene["boxes"] to be safe, or map it.
        # But User Prompt says: "self.active_box.set_world_poses(final_pos, quat)"
        # Use self.scene["boxes"] instead to avoid Attribute Error.
        
        # Global Indices Calculation
        env_indices = torch.arange(self.num_envs, device=self.device)
        global_active_idx = env_indices * self.cfg.max_boxes + self.box_idx
        
        # Write to Sim
        # Note: 'Quat' arg was a typo in user prompt, standard Isaac Lab uses (pos, rot) or (root_pos, root_rot)
        # Previous code used (final_pos, final_rot). 
        # But here 'quat' is local variable.
        # Isaac Lab View: write_root_pose_to_sim(root_pos_w, root_quat_w) usually.
        self.scene["boxes"].write_root_pose_to_sim(final_pos, quat, indices=global_active_idx)
        
        zeros_vel6 = torch.zeros((self.num_envs, 6), device=self.device)
        self.scene["boxes"].write_root_velocity_to_sim(zeros_vel6, indices=global_active_idx)
        
        # Save state
        self.last_target_pos = final_pos # Actually target_pos before holding override?
        # Stability check needs the intended target.
        # Regnerate target pos tensor for reference
        intended_pos = torch.zeros_like(final_pos)
        intended_pos[:, 0] = target_x
        intended_pos[:, 1] = target_y
        intended_pos[:, 2] = 0.0 # Floor reference? Or drop height?
        # Actually for drift check we just need XY of target.
        self.last_target_pos = intended_pos

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


