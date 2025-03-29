import torch
import torch.nn as nn

class MTN(nn.Module):
    def __init__(self):
        super(self).__init__()

        # First residual encoder before entering the HG cascade.
        # 256x256 -> 64x64 preprocessing
    
    def forward(self, imgs):
        x = self.enc(imgs)