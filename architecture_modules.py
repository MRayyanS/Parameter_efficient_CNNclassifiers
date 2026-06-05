
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

from utils import *




# ============================================================================
#  Some basic CNN modules used throughout
# ============================================================================


# Bottleneck convolutional blocks (without depthwise separable comnvolutions) 
""" 
- features are lifted to high dimension and shrunk back to fewer channels by pointwise convolutions
- Extremely useful in early layers of the network, keeps the model lean
"""

class ExpCompModule(nn.Module):
    def __init__(self, in_ch, middle_ch):
        super(ExpCompModule, self).__init__()

        self.Expand = nn.Sequential(
            nn.Conv2d(in_ch, middle_ch, kernel_size=3, padding=1), nn.BatchNorm2d(middle_ch)
        )

        self.Compress = nn.Sequential(
            nn.ReLU(), nn.Conv2d(middle_ch, in_ch, kernel_size=1), nn.BatchNorm2d(in_ch)
        )

    def forward(self, x):
        out = self.Expand(x)
        out = self.Compress(out)
        return out

class ExpCompBlock(nn.Module):
    def __init__(self, in_ch, middle_ch, num_modules=5):
        super(ExpCompBlock, self).__init__()
        
        # Create a list of modules
        self.res_modules = nn.ModuleList([
            ExpCompModule(in_ch, middle_ch) for _ in range(num_modules)
        ])
    
    def forward(self, x):
        for res_module in self.res_modules:
            x = res_module(x)
        return x


# Bottleneck modules with depthwise separable convolutions for residual learning
class DepthSepModule(nn.Module):
    def __init__(self, in_ch, middle_ch):
        super(DepthSepModule, self).__init__()

        # Depthwise Separable convolutional module
        self.conv_dws = nn.Sequential(
            nn.Conv2d(in_ch, middle_ch, kernel_size=1), nn.BatchNorm2d(middle_ch),
            nn.Conv2d(middle_ch, middle_ch, kernel_size=3, padding=1, groups=middle_ch), nn.BatchNorm2d(middle_ch),
            nn.GELU(),
            nn.Conv2d(middle_ch, in_ch, kernel_size=1)
        )

    def forward(self, x):
        x = self.conv_dws(x) + x
        return x

class DepthSepBlock(nn.Module):
    def __init__(self, in_ch, middle_ch, num_modules):
        super(DepthSepBlock, self).__init__()

        self.res_modules = nn.ModuleList([
            DepthSepModule(in_ch, middle_ch) for _ in range(num_modules)
        ])

    def forward(self, x):
        for module in self.res_modules:
            x = module(x)
        return x


# ConvNext like modules
class ConvNext_module(nn.Module):
    def __init__(self, in_ch_dim, kernel_dim, mlp_middle_dim):
        super().__init__()

        self.normalize_conv  = nn.BatchNorm2d(in_ch_dim)
        self.conv_head = nn.Sequential(
        nn.Conv2d(in_ch_dim, 2*in_ch_dim, kernel_size=1),
        nn.Conv2d(2*in_ch_dim, 2*in_ch_dim, kernel_size=kernel_dim, padding='same', groups=2*in_ch_dim ),
        nn.ReLU(),
        nn.Conv2d(2*in_ch_dim, in_ch_dim, kernel_size=1)
        )
        
        self.normalize_mlp  = nn.BatchNorm2d(in_ch_dim)
        self.mlp_head   = nn.Sequential( 
            nn.Conv2d(in_ch_dim, mlp_middle_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mlp_middle_dim, in_ch_dim, kernel_size=1) 
            )

    def forward(self, x):
        x = self.conv_head(self.normalize_conv(x)) + x
        x = self.mlp_head(self.normalize_mlp(x)) + x
        return x

class ConvNext_Blocks(nn.Module):
    def __init__(self, ch_dimensions: list, num_blocks: int):
        """
        Args:
            ch_dimensions: [in_ch_dim, kernel_size_dim, mlp_middle_dim]
        """
        super().__init__()

        self.in_ch_dim, self.kernel_size_dim, self.mlp_middle_dim = ch_dimensions
        self.convnext_modules = nn.ModuleList([
            ConvNext_module(self.in_ch_dim, self.kernel_size_dim, self.mlp_middle_dim) for _ in range(num_blocks)
        ])


    def forward(self, xin):
        x = xin
        for module in self.convnext_modules:
            x = module(x)
        return x + xin





# Multi head Attention Transformer modules
# classical attention based transformers
class MH_Vision_Attention_module(nn.Module):
    def __init__(self, att_dimensions: list):
        super().__init__()

        self.token_dim, self.att_emb_dim, self.att_out_dim, self.num_att_heads = att_dimensions

        self.temperature = nn.Parameter(torch.ones(self.num_att_heads, 1, 1))

        self.WQ = nn.Conv2d(self.token_dim, self.att_emb_dim * self.num_att_heads, kernel_size=1, bias=False)
        self.WK = nn.Conv2d(self.token_dim, self.att_emb_dim * self.num_att_heads, kernel_size=1, bias=False)
        self.WV = nn.Conv2d(self.token_dim, self.att_out_dim * self.num_att_heads, kernel_size=1, bias=False)

        self.mixing = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(self.att_out_dim * self.num_att_heads, self.token_dim, kernel_size=1, bias=False)
        )

    def forward(self, xin):
        B, C, H, W = xin.shape

        Q = self.WQ(xin).view(B, self.num_att_heads, self.att_emb_dim, -1)
        K = self.WK(xin).view(B, self.num_att_heads, self.att_emb_dim, -1)
        V = self.WV(xin).view(B, self.num_att_heads, self.att_out_dim, -1)

        attention = K.transpose(-2, -1) @ Q / (self.temperature * (self.att_emb_dim ** 0.5) )
        attention = F.softmax(attention, dim=-1)

        output = V @ attention
        output = output.view(B, self.num_att_heads * self.att_out_dim, H, W)

        output = self.mixing(output)
        return output

# multi head attention blocks
class MHA_Vision_Transformer_module(nn.Module):
    def __init__(self, att_dimensions: list, mlp_dimensions: list):
        """
        Args:
            att_dimensions: [token_dim, att_emb_dim, att_out_dim, num_att_heads]
            mlp_dimensions: [token_dim, mlp_middle_dim]
        """
        super().__init__()

        self.token_dim, self.att_emb_dim, self.att_out_dim, self.num_att_heads = att_dimensions
        self.mlp_middle_dim = mlp_dimensions[1]

        # normalization ---> multiple attention heads ---> mixing
        self.normalize_att = nn.BatchNorm2d(self.token_dim)
        self.multi_head_attention = MH_Vision_Attention_module(att_dimensions=att_dimensions)

        # normalization ---> mlp head
        self.normalize_mlp = nn.BatchNorm2d(self.token_dim)
        self.mlp_head = nn.Sequential(
            nn.Conv2d(self.token_dim, self.mlp_middle_dim, kernel_size=1), nn.GELU(), nn.Conv2d(self.mlp_middle_dim, self.token_dim, kernel_size=1)
            )

    
    def forward(self, xin):
        # forward pass for normalization ---> attention head
        x = self.multi_head_attention(self.normalize_att(xin)) + xin

        # forward pass for the mlp head
        x = self.mlp_head(self.normalize_mlp(x)) + x
        return x + xin
    

class MHA_Vision_Transformer_Blocks(nn.Module):
    def __init__(self, att_dimensions: list, mlp_dimensions: list, num_modules: int):
        """
        Args:
            att_dimensions: [token_dim, att_emb_dim, att_out_dim, num_att_heads]
            mlp_dimensions: [token_dim, mlp_middle_dim]
            num_modules: number of transformer modules
        """
        super().__init__()

        self.mha_transformer_modules = nn.ModuleList([
            MHA_Vision_Transformer_module(att_dimensions, mlp_dimensions) for _ in range(num_modules)
        ])

    def forward(self, x):
        for module in self.mha_transformer_modules:
            x = module(x)
        return x
