import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

from utils import *
from architecture_modules import *





# ============================================================================
#  CNN models for fashionMNIST
# ============================================================================


class fMNIST_train(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
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

class fMNIST_35k(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
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

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.resblock(x)
        
        x = self.conv_final(x)
        x = x.view(x.size(0), -1)
        return x



class fMNIST_90k(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
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

class CIFAR10_train(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 4, kernel_size=1), nn.BatchNorm2d(4), nn.ReLU()
        )
        # spatial dim = 32x32 
        self.conv2 = ExpCompBlock(4, 64, 6)
        
        # spatial dim = 32x32 
        self.conv3 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), # 16x16
            nn.Conv2d(64, 8, kernel_size=3), nn.BatchNorm2d(8), nn.ReLU(),             # 14x14
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)  # 7x7
        )
        
        self.resblock = MHA_Vision_Transformer_Blocks(att_dimensions=[16, 8, 16, 2], mlp_dimensions=[16, 64], num_modules=8)

        self.conv_final = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(),   # 5x5
            nn.Conv2d(64, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(),  # 3x3
            nn.Conv2d(256, num_classes, kernel_size=1), # 1x1
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
    

class CIFAR10_Transformer_88k(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 4, kernel_size=1), nn.BatchNorm2d(4), nn.ReLU()
        )
        # spatial dim = 32x32 
        self.conv2 = ExpCompBlock(4, 64, 6)
        
        # spatial dim = 32x32 
        self.conv3 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), # 16x16
            nn.Conv2d(64, 8, kernel_size=3), nn.BatchNorm2d(8), nn.ReLU(),             # 14x14
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)  # 7x7
        )
        
        self.resblock = MHA_Vision_Transformer_Blocks(att_dimensions=[16, 8, 16, 2], mlp_dimensions=[16, 64], num_modules=8)

        self.conv_final = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(),   # 5x5
            nn.Conv2d(64, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(),  # 3x3
            nn.Conv2d(256, num_classes, kernel_size=1), # 1x1
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
class CIFAR10_conv_45k(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 4, kernel_size=1), nn.BatchNorm2d(4), nn.ReLU()
        )
        # spatial dim = 32x32 
        self.conv2 = ExpCompBlock(4, 64, 2)
        
        # spatial dim = 32x32 
        self.conv3 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), # 16x16
            nn.Conv2d(64, 8, kernel_size=3), nn.BatchNorm2d(8), nn.ReLU(),    # 14x14
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)  # 7x7
        )
        
        self.resblock = DepthSepBlock(16, 64, 4)
        
        self.conv_final = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(),   # 5x5
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
    


## Best mid-size model -------------------------------------------------------------------------
class CIFAR10_conv_78k(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
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
            nn.Conv2d(16, 128, kernel_size=3), nn.BatchNorm2d(128), nn.ReLU(),   # 5x5
            nn.Conv2d(128, num_classes, kernel_size=3), nn.BatchNorm2d(num_classes), # 3x3
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
    


# best slighty bigger model -------------------------------------------------------------------------
class CIFAR10_conv_207k(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 8, kernel_size=1), nn.BatchNorm2d(8), nn.ReLU()
        )
        # spatial dim = 32x32 
        self.conv2 = ExpCompBlock(8, 128, 6)
        
        # spatial dim = 32x32 
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), # 16x16
            nn.Conv2d(64, 16, kernel_size=3), nn.BatchNorm2d(16), nn.ReLU(),             # 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)  # 7x7
        )
        
        self.resblock = DepthSepBlock(32, 128, 8)
        
        self.conv_final = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(),   # 5x5
            nn.Conv2d(64, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(),  # 3x3
            nn.Conv2d(256, num_classes, kernel_size=1), # 1x1
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




# ============================================================================
#  CNN models for CIFAR100
# ============================================================================

class CIFAR100_train(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), # 32x32
            nn.Conv2d(128, 4, kernel_size=1), nn.BatchNorm2d(4), nn.ReLU(), # 32x32
            ExpCompBlock(4, 128, 4)  # 32x32
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), # 16x16
            nn.Conv2d(128, 8, kernel_size=1), nn.BatchNorm2d(8), nn.ReLU(),    # 16x16
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
            nn.Conv2d(16, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU() # 6x6
        )
        
        self.resblock = DepthSepBlock(32, 128, 10)  # 6x6
        
        self.conv_final = nn.Sequential(
            nn.Conv2d(32, 512, kernel_size=1), nn.BatchNorm2d(512), nn.ReLU(),  # 6x6
            nn.Conv2d(512, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(),  # 6x6
            nn.Conv2d(256, num_classes, kernel_size=1), # 6x6
            nn.AdaptiveAvgPool2d(1)
        )
        
    def forward(self, x):
        # 1. Feature extraction and refinement
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.resblock(x)
        x = self.conv_final(x)
        x = torch.flatten(x, 1)
        return x



class CIFAR100_conv_326k(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), # 32x32
            nn.Conv2d(128, 4, kernel_size=1), nn.BatchNorm2d(4), nn.ReLU(), # 32x32
            ExpCompBlock(4, 128, 4)
        )

        # spatial dim = # 32x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), # 16x16
            nn.Conv2d(128, 16, kernel_size=1), nn.BatchNorm2d(16), nn.ReLU(),    # 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
            nn.Conv2d(32, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU() # 6x6
        )

        # spatial dim = 6x6
        self.resblock = DepthSepBlock(64, 128, 10)
        
        # spatial dim = 6x6
        self.conv_final = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=1), nn.BatchNorm2d(512), nn.ReLU(),  # 6x6
            nn.Conv2d(512, num_classes, kernel_size=1), # 6x6
            nn.AdaptiveAvgPool2d(1)
        )
        
    def forward(self, x):
        # 1. Feature extraction and refinement
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.resblock(x)
        x = self.conv_final(x)
        x = torch.flatten(x, 1)
        return x


class CIFAR100_277k(nn.Module):
    def __init__(self, num_classes=100):
        super(CIFAR100_277k, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), # 32x32
            nn.Conv2d(256, 4, kernel_size=1), nn.BatchNorm2d(4), nn.ReLU() # 32x32
        )

        # spatial dim = # 32x32
        self.conv2 = ExpCompBlock(4, 256, 5)

        # spatial dim = 16x16
        self.conv3 = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), # 16x16
            nn.Conv2d(128, 32, kernel_size=1), nn.BatchNorm2d(32), nn.ReLU(),    # 16x16
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
            nn.Conv2d(32, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU() # 6x6
        )

        # spatial dim = 6x6
        self.resblock = DepthSepBlock(32, 128, 10)
        
        # spatial dim = 6x6
        self.conv_final = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3), nn.BatchNorm2d(128), nn.ReLU(),    # 4x4
            nn.AdaptiveAvgPool2d(1)
        )

        self.FClayer = nn.Sequential(
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p=0.25),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p=0.25),
            nn.Linear(128, num_classes)
        )

    
    def forward(self, xin):
        x = self.conv1(xin)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.resblock(x)
        x = self.conv_final(x)

        x = torch.flatten(x, 1) 
        x = self.FClayer(x)
        return x




class CIFAR100_293k(nn.Module):
    def __init__(self, num_classes=100):
        super(CIFAR100_293k, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), # 32x32
            nn.Conv2d(256, 4, kernel_size=1), nn.BatchNorm2d(4), nn.ReLU() # 32x32
        )

        # spatial dim = # 32x32
        self.conv2 = ExpCompBlock(4, 256, 5)

        # spatial dim = 16x16
        self.conv3 = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), # 16x16
            nn.Conv2d(128, 32, kernel_size=1), nn.BatchNorm2d(32), nn.ReLU(),    # 16x16
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
            nn.Conv2d(32, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU() # 6x6
        )

        # spatial dim = 6x6
        self.resblock = DepthSepBlock(32, 128, 10)
        
        # spatial dim = 6x6
        self.conv_final = nn.Sequential(
            nn.Conv2d(32, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(),    # 6x6
            nn.Conv2d(256, 256, kernel_size=3, groups=256), nn.BatchNorm2d(256), nn.ReLU(),    # 4x4
            nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2
        )

        # --- NEW: Grouped MLP Logic ---
        self.num_feat_groups = 4   # 4 = 2x2 of spatial features
        self.group_input_dim = 256
        
        # Final MLP
        self.MLP = nn.Sequential(
            nn.Linear(self.group_input_dim, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(192, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )
        

    def forward(self, xin):
        # 1. Feature extraction and refinement
        x = self.conv1(xin)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.resblock(x)
        x = self.conv_final(x)

        # 2. Split into 4-128 dimensional feature vectors
        # Move spatial dims to a list: [Batch, 128, 4] then Permute to [Batch, 4, 128]
        x = x.view(x.size(0), 256, 4).permute(0, 2, 1)

        # 3. Make ensemble of all the parallel MLPs
        # x[:, 0, :] is Top-Left, x[:, 1, :] is Top-Right, etc.
        logits = sum(self.MLP(x[:, i, :]) for i in range(self.num_feat_groups))
        
        return logits
    


