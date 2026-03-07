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

        # THE EXPAND COMPONENT: Now Depthwise Separable
        self.Expand = nn.Sequential(
            # 1. Pointwise Expansion: Increase channels from in_ch to middle_ch first
            nn.Conv2d(in_ch, middle_ch, kernel_size=1), nn.BatchNorm2d(middle_ch),
            
            # 2. Depthwise Convolution: Now you have 'middle_ch' number of 3x3 filters
            # groups=middle_ch ensures each of the middle_ch channels gets its own 3x3 filter
            nn.Conv2d(middle_ch, middle_ch, kernel_size=3, padding=1, groups=middle_ch), nn.BatchNorm2d(middle_ch)
        )

        self.Compress = nn.Sequential(
            nn.ReLU(), nn.Conv2d(middle_ch, in_ch, kernel_size=1), nn.BatchNorm2d(in_ch)
        )

    def forward(self, x):
        out = self.Expand(x)
        out = self.Compress(out) + x 
        return out

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



# ============================================================================
#  CNN models for fashionMNIST
# ============================================================================


class FashionMNIST_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FashionMNIST_CNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), # 28x28
            nn.Conv2d(128, 2, kernel_size=1), nn.BatchNorm2d(2), nn.ReLU()
        )
        
        self.conv2 = ExpCompBlock(2, 64, 4)

        self.conv3 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14
            nn.Conv2d(64, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),             # 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),     # 7x7
        )
        
        self.resblock = DepthSepBlock(32, 128, 6)
        
        self.conv_final = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, num_classes, kernel_size=3)
        )

        
    def forward(self, xin):
        x = self.conv1(xin)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.resblock(x)
        
        x = self.conv_final(x)
        x = x.view(x.size(0), -1)
        return x



## Best tiny model -------------------------------------------------------------------------

class FashionMNIST_35k(nn.Module):
    def __init__(self, num_classes=10):
        super(FashionMNIST_35k, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), # 28x28
            nn.Conv2d(64, 2, kernel_size=1), nn.BatchNorm2d(2), nn.ReLU()
        )
        
        self.conv2 = ExpCompBlock(2, 64, 4)

        self.conv3 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14
            nn.Conv2d(64, 8, kernel_size=3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),             # 14x14
            nn.Conv2d(8, 8, kernel_size=3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),     # 7x7
        )
        
        self.resblock = DepthSepBlock(8, 64, 8)
        
        self.conv_final = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=3)
        )

        
    def forward(self, xin):
        x = self.conv1(xin)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.resblock(x)
        
        x = self.conv_final(x)
        x = x.view(x.size(0), -1)
        return x



class FashionMNIST_90k(nn.Module):
    def __init__(self, num_classes=10):
        super(FashionMNIST_90k, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), # 28x28
            nn.Conv2d(128, 2, kernel_size=1), nn.BatchNorm2d(2), nn.ReLU()
        )
        
        self.conv2 = ExpCompBlock(2, 64, 4)

        self.conv3 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14
            nn.Conv2d(64, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),             # 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),     # 7x7
        )
        
        self.resblock = DepthSepBlock(32, 128, 6)
        
        self.conv_final = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, num_classes, kernel_size=3)
        )

        
    def forward(self, xin):
        x = self.conv1(xin)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.resblock(x)
        
        x = self.conv_final(x)
        x = x.view(x.size(0), -1)
        return x



# ============================================================================
#  CNN models for CIFAR10
# ============================================================================

class CIFAR10_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10_CNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), # 32x32
            nn.Conv2d(128, 4, kernel_size=1), nn.BatchNorm2d(4), nn.ReLU() # 32x32
        )

        # spatial dim = # 32x32
        self.conv2 = ExpCompBlock(4, 128, 5)

        # spatial dim = 16x16
        self.conv3 = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), # 16x16
            nn.Conv2d(128, 16, kernel_size=1), nn.BatchNorm2d(16), nn.ReLU(),    # 16x16
            nn.Conv2d(16, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
            nn.Conv2d(64, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU() # 6x6
        )

        # spatial dim = 6x6
        self.resblock = DepthSepBlock(32, 128, 10)
        
        # spatial dim = 6x6
        self.conv_final = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(),   # 4x4
            nn.Conv2d(64, num_classes, kernel_size=3), nn.BatchNorm2d(num_classes), # 2x2
            nn.AdaptiveAvgPool2d(1)
        )

    
    def forward(self, xin):
        x = self.conv1(xin)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.resblock(x)
        x = self.conv_final(x)

        x = torch.flatten(x, 1) 
        return x



## Best tiny model -------------------------------------------------------------------------

class CIFAR10_56k(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10_56k, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 4, kernel_size=1), nn.BatchNorm2d(4), nn.ReLU()
        )
        # spatial dim = 32x32 
        self.conv2 = ExpCompBlock(4, 64, 4)
        
        # spatial dim = 32x32 
        self.conv3 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), # 16x16
            nn.Conv2d(64, 8, kernel_size=3), nn.BatchNorm2d(8), nn.ReLU(),    # 14x14
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)  # 7x7
        )
        
        self.resblock = DepthSepBlock(16, 64, 8)
        
        self.conv_final = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU(),   # 5x5
            nn.Conv2d(32, num_classes, kernel_size=3), nn.BatchNorm2d(num_classes), # 3x3
            nn.AdaptiveAvgPool2d(1)
        )

    
    def forward(self, xin):
        x = self.conv1(xin)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.resblock(x)
        x = self.conv_final(x)

        x = torch.flatten(x, 1) 
        return x


## Best decent size model -------------------------------------------------------------------------

class CIFAR10_150k(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10_150k, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 4, kernel_size=1), nn.BatchNorm2d(4), nn.ReLU()
        )
        # spatial dim = 32x32 
        self.conv2 = ExpCompBlock(4, 128, 4)

        # spatial dim = 32x32 
        self.conv3 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), # 16x16
            nn.Conv2d(64, 16, kernel_size=3), nn.BatchNorm2d(16), nn.ReLU(),    # 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)  # 7x7
        )
        
        self.resblock = DepthSepBlock(32, 128, 8)
        
        self.conv_final = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(),   # 5x5
            nn.Conv2d(64, num_classes, kernel_size=3), nn.BatchNorm2d(num_classes), # 3x3
            nn.AdaptiveAvgPool2d(1)
        )

    
    def forward(self, xin):
        x = self.conv1(xin)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.resblock(x)
        x = self.conv_final(x)

        x = torch.flatten(x, 1) 
        return x







