
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
        self.x_head = nn.Linear(hidden, self.L)
        self.y_head = nn.Linear(hidden, self.W)
        self.value = nn.Linear(hidden, 1)

    def forward(self, enc_feat, mask=None, gating_lambda=2.0):
        h = self.flatten(enc_feat)
        h = self.trunk(h)
        logits_pick = self.pick(h)
        logits_yaw = self.yaw(h)
        logits_x = self.x_head(h)
        logits_y = self.y_head(h)
        if mask is not None:
            eps = 1e-6
            msum_x = (mask.sum(dim=2).clamp(min=eps))
            logits_x = logits_x + gating_lambda * msum_x.log()
            mmean_y = (mask.mean(dim=1).clamp(min=eps))
            logits_y = logits_y + gating_lambda * mmean_y.log()
        value = self.value(h).squeeze(-1)
        return logits_pick, logits_yaw, logits_x, logits_y, value
