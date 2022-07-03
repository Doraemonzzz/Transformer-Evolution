import torch
import torch.nn as nn

class TokenShift(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

    def forward(self, x):
        d = x.size()[-1]
        x = torch.cat([self.time_shift(x[..., :d // 2]), x[..., d // 2:]], dim=-1)
        
        return x
