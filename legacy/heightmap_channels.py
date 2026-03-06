
import numpy as np
from scipy.ndimage import uniform_filter

def compute_channels(height_cm, occ, box_l_cells, box_w_cells, density_proj, stiffness_proj,
                     safety_margin_cells=2, h_cap_mask=None, kernel=3):
    """
    Compute proxy channels for the 2D height-map.
    Inputs:
      - height_cm: (L,W) float32 heights in cm
      - occ: (L,W) uint8 occupancy (1 if filled up to H)
      - box_l_cells, box_w_cells: int, current box base in cells for Support estimation
      - density_proj, stiffness_proj: (L,W) float32 projections
      - h_cap_mask: (L,W) bool, 1 if H < Hmax - h_box else 0 (allowed), else None
      - kernel: local window size for smoothing/variance (3 or 5)
    Returns dict of channels as float32 arrays normalized to [0,1] where appropriate.
    """
    L, W = height_cm.shape
    k = kernel if kernel in (3,5) else 3

    H = height_cm.astype(np.float32)
    H_norm = H / max(1.0, float(np.max(H)) if np.max(H)>0 else 1.0)

    Occ = occ.astype(np.float32)

    wx = max(1, int(box_l_cells))
    wy = max(1, int(box_w_cells))
    pad_x = wx//2; pad_y = wy//2
    padded = np.pad(Occ, ((pad_x,pad_x),(pad_y,pad_y)), mode='edge')
    Support = np.zeros_like(Occ, dtype=np.float32)
    for i in range(L):
        for j in range(W):
            sx = i; sy = j
            window = padded[sx:sx+wx, sy:sy+wy]
            Support[i,j] = float(np.mean(window))

    H_blur = uniform_filter(H, size=k, mode='nearest')
    H2_blur = uniform_filter(H*H, size=k, mode='nearest')
    var_local = np.maximum(0.0, H2_blur - H_blur*H_blur)
    Roughness = var_local / max(1e-6, float(np.max(var_local)) if np.max(var_local)>0 else 1e-6)

    def minmax01(A):
        a_min = float(np.min(A)); a_max = float(np.max(A))
        if a_max - a_min < 1e-9:
            return np.zeros_like(A, dtype=np.float32)
        return (A - a_min)/(a_max - a_min)

    DensityP = minmax01(density_proj.astype(np.float32))
    StiffnessP = minmax01(stiffness_proj.astype(np.float32))

    SM = np.zeros((L,W), dtype=np.float32)
    SM[:safety_margin_cells,:] = 1.0
    SM[-safety_margin_cells:,:] = 1.0
    SM[:,:safety_margin_cells] = 1.0
    SM[:,-safety_margin_cells:] = 1.0
    Safe = 1.0 - SM

    HeightCap = h_cap_mask.astype(np.float32) if h_cap_mask is not None else np.ones((L,W), dtype=np.float32)

    return {
        "H": H_norm.astype(np.float32),
        "Occ": Occ,
        "Support": np.clip(Support, 0.0, 1.0),
        "Roughness": np.clip(Roughness, 0.0, 1.0),
        "Density_proj": np.clip(DensityP, 0.0, 1.0),
        "Stiffness_proj": np.clip(StiffnessP, 0.0, 1.0),
        "SafetyMargin": Safe,
        "HeightCap": HeightCap,
    }
