
import torch
import torch.nn as nn

class PolicyHeads(nn.Module):
    def __init__(self, enc_out_ch:int, grid_hw, n_pick:int, n_yaw:int, hidden:int=256):
        super().__init__()
        self.L, self.W = grid_hw
        # Encoder2D downsamples by 4 (stride 4 total), not 8
        feat = enc_out_ch * (self.L//4) * (self.W//4)
        self.flatten = nn.Flatten()
        self.trunk = nn.Sequential(nn.Linear(feat, hidden), nn.ReLU(inplace=True))
        self.pick = nn.Linear(hidden, n_pick)
        self.yaw = nn.Linear(hidden, n_yaw)
        # Joint position head
        self.pos_head = nn.Linear(hidden, self.L * self.W)
        self.value = nn.Linear(hidden, 1)

    def forward(self, enc_feat, mask=None, gating_lambda=2.0):
        h = self.flatten(enc_feat)
        h = self.trunk(h)
        logits_pick = self.pick(h)
        logits_yaw = self.yaw(h)
        logits_pos = self.pos_head(h)
        
        if mask is not None:
            # mask is (B, L*W)
            # logits_pos is (B, L*W)
            # Apply -inf where mask is 0 (False)
            logits_pos = logits_pos.masked_fill(mask == 0, -float('inf'))
            
        value = self.value(h).squeeze(-1)
        return logits_pick, logits_yaw, logits_pos, value
