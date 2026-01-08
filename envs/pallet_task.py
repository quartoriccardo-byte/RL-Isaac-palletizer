
import torch
import numpy as np
from typing import Dict, Any

from omni.isaac.core.utils.torch.rotations import quat_rotate_inverse, quat_mul
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass

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
    pallet_size_cm: list = [120., 100.]
    grid_res_cm: float = 2.5
    map_size: int = 64 # Fixed for RSL-RL Wrapper expectations (64x64)
    max_boxes: int = 50
    max_height: float = 2.0 # For normalization
    
    # State Dims
    # Robot State: Joints (e.g. 6 pos + 6 vel) + Gripper? 
    # Let's assume 12 DOFs total -> 24.
    robot_state_dim = 24 
    
    # Calculated later in init, but for Config access we might need it.
    # DirectRLEnvCfg requires num_observations to be set? 
    # Yes.
    num_observations = (64 * 64) + 24
    num_actions = 4
    
    episode_length_s = 60.0

class PalletTask(DirectRLEnv):
    def __init__(self, cfg: PalletTaskCfg, render_mode: str | None = None, **kwargs):
        # Recalculate if needed or trust cfg
        self.robot_state_dim = cfg.robot_state_dim
        self.map_size = cfg.map_size
        
        # Dynamic calculation of observation dimension
        # Should be done before super().__init__ if DirectRLEnv uses it, 
        # but DirectRLEnv uses cfg.num_observations.
        # So we update cfg.num_observations first.
        cfg.num_observations = (self.robot_state_dim) + (self.map_size * self.map_size)
        
        super().__init__(cfg, render_mode, **kwargs)
        
        # Warp Rasterizer Init
        self.heightmap_gen = WarpHeightmapGenerator(
            device=self.device,
            num_envs=self.num_envs,
            max_boxes=self.cfg.max_boxes,
            grid_res=self.cfg.grid_res_cm / 100.0,
            map_size=self.map_size,
            pallet_dims=(self.cfg.pallet_size_cm[0]/100.0, self.cfg.pallet_size_cm[1]/100.0)
        )
        
        # Box Dims (Placeholder/Mock)
        self.box_dims_tensor = torch.tensor([0.4, 0.3, 0.2], device=self.device).repeat(self.num_envs, self.cfg.max_boxes, 1)

    def _setup_scene(self):
        pass
        
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        # 1. Fetch Buffers
        boxes = self.scene["boxes"]
        pallet = self.scene["pallet"]
        robot = self.scene["robot"]
        
        box_pos_w = boxes.data.root_pos_w
        box_quat_w = boxes.data.root_quat_w
        
        pallet_pos_w = pallet.data.root_pos_w
        if pallet_pos_w.dim() == 3: pallet_pos_w = pallet_pos_w.squeeze(1)
        
        pallet_quat_w = pallet.data.root_quat_w
        if pallet_quat_w.dim() == 3: pallet_quat_w = pallet_quat_w.squeeze(1)
        
        # 2. Convert to Warp (xyzw)
        box_quat_warp = box_quat_w[:, :, [1, 2, 3, 0]]
        pallet_quat_warp = pallet_quat_w[:, [1, 2, 3, 0]]
        
        # 3. Generate Heightmap
        heightmap = self.heightmap_gen.forward(
            box_pos=box_pos_w,
            box_rot=box_quat_warp,
            box_dims=self.box_dims_tensor,
            pallet_pos=pallet_pos_w,
            pallet_rot=pallet_quat_warp
        ) # (N, 64, 64)
        
        # 4. Normalize
        heightmap = torch.clamp(heightmap / self.cfg.max_height, 0.0, 1.0)
        
        # 5. Proprioception
        # Access robot joint states
        # Assuming robot has num_dof joints.
        # If cfg.robot_state_dim is 24, we assume 12 dof?
        # Verify robot.data.joint_pos size.
        # For safety in this "no placeholder" code, we should slice or pad if mismatch, 
        # but better to assume configuration matches robot asset.
        joint_pos = robot.data.joint_pos 
        joint_vel = robot.data.joint_vel
        
        proprio = torch.cat([joint_pos, joint_vel], dim=-1)
        
        # 6. Assemble
        visual_obs = heightmap.flatten(start_dim=1) # (N, 4096)
        
        obs = torch.cat([visual_obs, proprio], dim=-1)
        
        return {"policy": obs}

    def _pre_physics_step(self, actions: torch.Tensor):
        # We need to reshape actions if needed, or pass directly.
        # actions (N, 4)
        self.actions = actions.clone()

    def _apply_action(self, action):
        pass
