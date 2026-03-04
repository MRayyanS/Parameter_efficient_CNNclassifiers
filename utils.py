import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F


# ============================================================================
# Build Model - CNN based
# ============================================================================

# canonical vanilla Conv2d block and modules like in VGGnet
class ConvModule(nn.Module):
    def __init__(self, in_ch):
        super(ConvModule, self).__init__()

        self.convmodule = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1), nn.BatchNorm2d(in_ch), nn.ReLU()
        )

    def forward(self, x):
        x = self.convmodule(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, num_blocks):
        super(ConvBlock, self).__init__()

        self.conv_modules = nn.ModuleList([
            ConvModule(in_ch) for _ in range(num_blocks)
        ])

    def forward(self, x):
        for module in self.conv_modules:
            x = module(x)
        return x


# residual blocks like original ResNet paper
class VanillaResModule(nn.Module):
    def __init__(self, in_ch):
        super(VanillaResModule, self).__init__()

        self.VanResModule = nn.Sequential(
            nn.ReLU(), nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1), nn.BatchNorm2d(in_ch)
        )

    def forward(self, x):
        x = self.VanResModule(x) + x
        return x

class VanillaResBlock(nn.Module):
    def __init__(self, in_ch, num_modules):
        super(VanillaResBlock, self).__init__()

        self.resModules = nn.ModuleList([
            VanillaResModule(in_ch) for _ in range(num_modules)
        ])

    def forward(self, x):
        for module in self.resModules:
            x = module(x)
        return x



# Blocks with convolutional feature channels expand and shrink
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

# custom residual blocks with configurable number of modules
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



# Custom expand-compress modules wit residual connections
class ExpCompResModule(nn.Module):
    def __init__(self, in_ch, middle_ch):
        super(ExpCompResModule, self).__init__()

        # THE EXPAND COMPONENT
        self.Expand = nn.Sequential(
            # 1. Pointwise Expansion: Increase channels from in_ch to middle_ch first
            nn.Conv2d(in_ch, middle_ch, kernel_size=3, padding=1), nn.BatchNorm2d(middle_ch)
        )

        self.Compress = nn.Sequential(
            nn.ReLU(), nn.Conv2d(middle_ch, in_ch, kernel_size=1), nn.BatchNorm2d(in_ch)
        )

    def forward(self, x):
        out = self.Expand(x)
        out = self.Compress(out) + x 
        return out

# custom residual blocks with configurable number of modules
class ExpCompResBlock(nn.Module):
    def __init__(self, in_ch, middle_ch, num_modules=5):
        super(ExpCompResBlock, self).__init__()
        
        # Create a list of modules
        self.res_modules = nn.ModuleList([
            ExpCompResModule(in_ch, middle_ch) for _ in range(num_modules)
        ])
    
    def forward(self, x):
        for res_module in self.res_modules:
            x = res_module(x)
        return x


# Depthwise separable modules and block
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




# custom residual modules
class DualResModule(nn.Module):
    def __init__(self, in_ch, middle_ch):
        super(DualResModule, self).__init__()

        self.Expand = nn.Sequential(
            nn.Conv2d(in_ch, middle_ch, kernel_size=3, padding=1), nn.BatchNorm2d(middle_ch)
        )

        self.Compress = nn.Sequential(
            nn.ReLU(), nn.Conv2d(middle_ch, in_ch, kernel_size=1), nn.BatchNorm2d(in_ch)
        )

    def forward(self, x, y=0):
        y = self.Expand(x)   + y
        x = self.Compress(y) + x
        return x, y


# custom residual blocks with configurable number of modules
class DualResBlock(nn.Module):
    def __init__(self, in_ch, middle_ch, num_modules=5):
        super(DualResBlock, self).__init__()
        
        # Create a list of modules
        self.res_modules = nn.ModuleList([
            DualResModule(in_ch, middle_ch) for _ in range(num_modules)
        ])
    
    def forward(self, x, y=0):
        for res_module in self.res_modules:
            x, y = res_module(x, y)
        return x, y






# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_loss_curves(train_loss_history, val_loss_history, batches_per_epoch):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    train_x = np.arange(len(train_loss_history))
    val_x = np.arange(1, len(val_loss_history) + 1) * batches_per_epoch - 1
    
    # Plotting
    ax.plot(train_x, train_loss_history, label='Training Loss', alpha=0.7, linewidth=0.8)
    ax.plot(val_x, val_loss_history, label='Validation Loss', 
            marker='o', markersize=5, linewidth=2, color='red')
    
    ax.set_xlabel('Mini-batch Number', fontsize=12, labelpad=5)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- SECONDARY X-AXIS FOR EPOCHS ---
    ax2 = ax.secondary_xaxis('bottom', functions=(lambda x: x, lambda x: x))
    ax2.spines['bottom'].set_position(('outward', 35)) 
    
    # Generate ticks every 10 epochs
    # We use step=10 in arange, and multiply by batches_per_epoch to find the x-position
    total_epochs = len(val_loss_history)
    epoch_indices = np.arange(0, total_epochs + 1, 10)
    
    # Ensure the last epoch is always included if it's not a multiple of 10
    if total_epochs not in epoch_indices:
        epoch_indices = np.append(epoch_indices, total_epochs)
        
    epoch_ticks = epoch_indices * batches_per_epoch
    
    ax2.set_xticks(epoch_ticks)
    ax2.set_xticklabels([str(i) for i in epoch_indices])
    ax2.set_xlabel('Epoch (Every 10)', fontsize=12)

    # Add vertical lines only at the 10-epoch marks to keep the plot clean
    for tick in epoch_ticks:
        ax.axvline(x=tick, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
    print("\n✓ Loss curve plot saved with 10-epoch intervals.")
    plt.show()


# ============================================================================
# Print the parameters of the model
# ============================================================================

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


# ============================================================================
# Guassian noise and blur for data augmentation
# ============================================================================

# Custom Gaussian Noise transform with dynamic std
class AddGaussianNoise:
    def __init__(self, mean=0., std=0.1, p=0.95):
        self.mean = mean
        self.std = std
        self.p = p
        self.current_std = std
    
    def set_std(self, std):
        self.current_std = std
    
    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn(tensor.size()) * self.current_std + self.mean
            return tensor + noise
        return tensor
    
    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.current_std}, p={self.p})'



# Custom Gaussian Blur transform with dynamic sigma
class AddGaussianBlur:
    def __init__(self, kernel_size=3, sigma=1.0, p=0.95):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
        self.current_sigma = sigma
    
    def set_sigma(self, sigma):
        self.current_sigma = sigma
    
    def __call__(self, img):
        if torch.rand(1).item() < self.p and self.current_sigma > 0:
            return transforms.functional.gaussian_blur(img, self.kernel_size, [self.current_sigma])
        return img
    
    def __repr__(self):
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.current_sigma}, p={self.p})'

