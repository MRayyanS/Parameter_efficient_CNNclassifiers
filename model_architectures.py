import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

from utils import *


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





class FashionMNIST_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FashionMNIST_CNN, self).__init__()
        
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






# ============================================================================
# Build Model - CNN based
# ============================================================================

class CIFAR10_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10_CNN, self).__init__()
        
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







