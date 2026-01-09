
import torch
import warp as wp
import numpy as np

# Initialize Warp
if not wp.is_initialized():
    wp.init()

@wp.kernel
def rasterize_heightmap_kernel(
    box_positions: wp.array(dtype=wp.vec3),      # (N_envs * max_boxes)
    box_orientations: wp.array(dtype=wp.quat),   # (N_envs * max_boxes)
    box_dimensions: wp.array(dtype=wp.vec3),     # (N_envs * max_boxes)
    pallet_positions: wp.array(dtype=wp.vec3),   # (N_envs)
    pallet_orientations: wp.array(dtype=wp.quat),# (N_envs)
    grid_map: wp.array(dtype=float, ndim=3),     # (N_envs, H, W)
    num_envs: int,
    max_boxes: int,
    grid_res: float,
    map_size_x: int, # Rows (corresponding to X axis usually if mapped u->x)
    map_size_y: int  # Cols (corresponding to Y axis usually if mapped v->y)
):
    """
    Ray-Free Projection-Based Rasterizer.
    Parallelized per pixel.
    tid = env_id * (SizeX * SizeY) + pixel_index
    
    Grid Map Dimensions: (N, map_size_x, map_size_y)
    u -> 0..map_size_x (Index 1)
    v -> 0..map_size_y (Index 2)
    """
    tid = wp.tid()
    pixels_per_env = map_size_x * map_size_y
    
    # 1. Determine Environment and Pixel Coordinates
    env_id = tid // pixels_per_env
    pixel_linear_idx = tid % pixels_per_env
    
    if env_id >= num_envs:
        return
        
    u = pixel_linear_idx // map_size_y # Row index (0..X-1) 
    v = pixel_linear_idx % map_size_y  # Col index (0..Y-1)
    
    # Note: Logic above implies grid_map is contiguous row-major: R0[0..W], R1[0..W]...
    # u is index for SizeX (Rows), v is index for SizeY (Cols).
    
    # 2. Calculate World Position of this Pixel P_world
    # Grid is centered on the Pallet.
    # Local Grid Coord (x, y).
    # map_size_x corresponds to spatial dimension X? 
    # Let's assume u -> X axis, v -> Y axis.
    
    grid_size_x = float(map_size_x) * grid_res
    grid_size_y = float(map_size_y) * grid_res
    
    # Pixel center in Grid Frame (Local)
    # u=0 -> -size_x/2 + res/2
    px_local_x = (float(u) + 0.5) * grid_res - (grid_size_x * 0.5)
    px_local_y = (float(v) + 0.5) * grid_res - (grid_size_y * 0.5)
    
    # Transform Pixel to World Frame
    pallet_pos = pallet_positions[env_id]
    pallet_quat = pallet_orientations[env_id]
    
    # Rotate pixel offset by pallet orientation
    p_offset_world = wp.quat_rotate(pallet_quat, wp.vec3(px_local_x, px_local_y, 0.0))
    p_world = pallet_pos + p_offset_world
    
    curr_max_h = 0.0 
    
    # 3. Iterate over every Box in this Environment
    base_idx = env_id * max_boxes
    
    for i in range(max_boxes):
        idx = base_idx + i
        
        dims = box_dimensions[idx]
        if dims[0] <= 0.0:
            continue
            
        b_pos = box_positions[idx]
        b_quat = box_orientations[idx]
        
        # 4. Transform P_world into Box Local Frame
        diff = p_world - b_pos
        # Inverse rotate
        p_box_local = wp.quat_rotate_inv(b_quat, diff)
        
        # 5. Check Intersection (Point in Box)
        half_l = dims[0] * 0.5
        half_w = dims[1] * 0.5
        
        # Check X/Y bounds in box frame (infinite prism check effectively)
        if (p_box_local[0] >= -half_l and p_box_local[0] <= half_l and
            p_box_local[1] >= -half_w and p_box_local[1] <= half_w):
            
            # Inside!
            # Simplified Z height: Top of box at box center Z. 
            # Valid for flat-stacked boxes.
            z_top = b_pos[2] + dims[2] * 0.5
            
            if z_top > curr_max_h:
                curr_max_h = z_top

    # Write Result
    grid_map[env_id, u, v] = curr_max_h

class WarpHeightmapGenerator:
    def __init__(self, device: str, num_envs: int, max_boxes: int, grid_res: float, map_size: tuple, pallet_dims: tuple):
        # map_size is now (size_x, size_y)
        # Ensure device is a string compatible with Warp
        if isinstance(device, torch.device):
            self.device = str(device)
        else:
            self.device = str(device)
            
        if "cuda" in self.device and ":" not in self.device:
             self.device = "cuda:0"
             
        self.num_envs = num_envs
        self.max_boxes = max_boxes
        self.grid_res = grid_res
        
        if isinstance(map_size, int):
            self.map_size_x = map_size
            self.map_size_y = map_size
        else:
            self.map_size_x = map_size[0]
            self.map_size_y = map_size[1]
            
        self.pallet_dims = pallet_dims
        
        # Allocate grid
        self.grid_wp = wp.zeros((num_envs, self.map_size_x, self.map_size_y), dtype=float, device=device)

    def forward(self, 
                box_pos: torch.Tensor, 
                box_rot: torch.Tensor, 
                box_dims: torch.Tensor, 
                pallet_pos: torch.Tensor,
                pallet_rot: torch.Tensor = None) -> torch.Tensor:
        """
        Generates heightmaps.
        Inputs: Torch Tensors on GPU.
        box_pos: (N, M, 3)
        box_rot: (N, M, 4) (x, y, z, w)
        box_dims: (N, M, 3)
        pallet_pos: (N, 3)
        pallet_rot: (N, 4)
        """
        
        self.grid_wp.zero_()
        
        # Conversion
        box_pos_wp = wp.from_torch(box_pos.contiguous(), dtype=wp.vec3)
        box_rot_wp = wp.from_torch(box_rot.contiguous(), dtype=wp.quat)
        box_dims_wp = wp.from_torch(box_dims.contiguous(), dtype=wp.vec3)
        pallet_pos_wp = wp.from_torch(pallet_pos.contiguous(), dtype=wp.vec3)
        
        if pallet_rot is None:
             # Identity quat (0, 0, 0, 1) to (0, 0, 0, 1) scalar real part? 
             # Warp/Isaac Lab usually (x,y,z,w). w real.
             # Identity is (0,0,0,1).
             ident = torch.zeros((self.num_envs, 4), device=box_pos.device)
             ident[:, 3] = 1.0 # w=1
             pallet_rot_wp = wp.from_torch(ident, dtype=wp.quat)
        else:
             pallet_rot_wp = wp.from_torch(pallet_rot.contiguous(), dtype=wp.quat)

        # Launch
        dim = self.num_envs * self.map_size_x * self.map_size_y
        
        wp.launch(
            kernel=rasterize_heightmap_kernel,
            dim=dim,
            inputs=[
                box_pos_wp,
                box_rot_wp,
                box_dims_wp,
                pallet_pos_wp,
                pallet_rot_wp,
                self.grid_wp,
                self.num_envs,
                self.max_boxes,
                self.grid_res,
                self.map_size_x,
                self.map_size_y
            ],
            device=self.device
        )
        
        return wp.to_torch(self.grid_wp)
