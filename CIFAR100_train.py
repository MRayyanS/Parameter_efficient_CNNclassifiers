import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import numpy as np
import copy, random, time


from utils import *
from model_architectures import *

# only to supress warning messages
import warnings
# This ignores the NumPy/Torchvision compatibility warning without needing the np attribute
warnings.filterwarnings("ignore", message=".*VisibleDeprecationWarning.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Essential global variables
# ============================================================================
np.random.seed(99)
torch.manual_seed(99)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# ============================================================================
# LOAD appropriate DATASET and create train/val split
# ============================================================================

DATA_mean = (0.5071, 0.4867, 0.4408)
DATA_std  = (0.2673, 0.2564, 0.2762)

# define noise and blur data augmentation objects
gaussian_noise = AddGaussianNoise(mean=0., std=0.05, p=0.5) 
gaussian_blur  = AddGaussianBlur(kernel_size=3, sigma=0.05, p=0.5)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(DATA_mean, DATA_std),
    gaussian_noise, 
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    gaussian_blur
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(DATA_mean, DATA_std),
])


# Load data
full_training_data_augmented = datasets.CIFAR100(
    root="./data", train=True, download=True, transform=train_transform
)

full_training_data_no_aug = datasets.CIFAR100(
    root="./data", train=True, download=True, transform=eval_transform
)

test_data = datasets.CIFAR100(
    root="./data", train=False, download=True, transform=eval_transform
)

# ----------------------------------------------------------------------------
# Create train/val split
num_classes = 100
train_samples_per_class = 500
val_samples_per_class = 0

class_indices = {i: [] for i in range(num_classes)}
labels = full_training_data_no_aug.targets

for idx, label in enumerate(labels):
    class_indices[label].append(idx)

for class_id in range(num_classes):
    # Shuffle the list in-place using random.shuffle (built-in)
    random.shuffle(class_indices[class_id])

train_indices = []
val_indices = []

for class_id in range(num_classes):
    train_indices.extend(class_indices[class_id][:train_samples_per_class])
    val_indices.extend(class_indices[class_id][train_samples_per_class:train_samples_per_class + val_samples_per_class])

np.random.shuffle(train_indices)
np.random.shuffle(val_indices)


# ----------------------------------------------------------------------------
# CREATE DATALOADERS

train_data = Subset(full_training_data_augmented, train_indices)
val_data = Subset(full_training_data_no_aug, val_indices)

batch_size = 256
train_loader = DataLoader(
    train_data, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=6,
    persistent_workers=True,
    prefetch_factor=2         # Workers will prepare 2 batches ahead
)
val_loader = DataLoader(
    test_data, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=6,
    persistent_workers=True
)
test_loader = DataLoader(
    test_data, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=6,
)


# ============================================================================
# Build train loop, validation, and test functions
# ============================================================================

def train(model, epoch, train_loss_history):
    model.train()
    epoch_loss = 0.0
    
    if epoch/num_epochs <= 0.7:
        gaussian_noise.set_std = 0.0125 * ( epoch/num_epochs )
        gaussian_blur.set_sigma = 0.0125 * ( epoch/num_epochs )

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # forward pass
        outputs = model(images)

        # Calculate Loss = CEntropy + L2 regularization
        CEloss = criterion(outputs, labels)
        loss   = CEloss

        # backprop and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # ------------------------------------        
        train_loss_history.append(CEloss.detach())
        epoch_loss += (CEloss.detach() - epoch_loss)/(i+1)

    # At the end of the train() function loop
    torch.mps.empty_cache()
    print(f'Epoch = {epoch}, Training Loss: {epoch_loss:.4f}')


# function for validation
def validate(model):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.float32)
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            probs = F.softmax(outputs, dim=1)
            
            for t in range(len(labels)):
                true_class = labels[t].long()
                confusion_matrix[true_class] += probs[t].cpu()
                class_counts[true_class] += 1
    
    accuracy = 100. * correct / total
    avg_loss = val_loss / len(val_loader)
    
    confusion_matrix_avg = confusion_matrix / class_counts.unsqueeze(1)
    
    return avg_loss, accuracy, confusion_matrix_avg.numpy()


def test(model):
    model.eval()
    print(f'Testing with the best model on the test data')
    correct = 0
    total = 0
    test_loss = 0.0
    
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.float32)
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            probs = F.softmax(outputs, dim=1)
            
            for t in range(len(labels)):
                true_class = labels[t].long()
                confusion_matrix[true_class] += probs[t].cpu()
                class_counts[true_class] += 1
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    print(f'Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}')
    
    confusion_matrix_avg = confusion_matrix / class_counts.unsqueeze(1)
    
    return avg_loss, accuracy, confusion_matrix_avg.numpy()



load_pretrained = True


# ============================================================================
# Training loop
# ============================================================================

if __name__ == '__main__':
    
    # define the model to be trained
    model = CIFAR100_train(num_classes=num_classes).to(device)

    if load_pretrained == True:
        checkpoint = torch.load('CIFAR100_model.pth', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    # count and print the number of parameters
    count_parameters(model)

    # define loss criterion to train the model
    criterion = nn.CrossEntropyLoss()

    # printing some basic info
    print(f'Starting training... using device: {device}')
    print(f"✓ Batch size: {batch_size}, Training batches per epoch: {len(train_loader)}")
    print(f"✓ Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}\n")
    
    num_epochs = 25
    learning_rate = 0.0005    # 0.0075
    regularizer = 0.001
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=regularizer)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=3, min_lr=5e-5)
    
    train_loss_history  = []
    val_loss_history    = []
    
    best_acc = 0.0
    for epoch in range(num_epochs):

        # 1. Train the model - and update the train loss and confusion matrix
        train(model, epoch, train_loss_history)

        # 2. evealuate on validation data and compute val_loss, val_acc, and confusion_matrix
        val_loss, val_acc, val_conf_matrix = validate(model)
        val_loss_history.append(val_loss)


        scheduler.step(val_loss)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_model_state_dict = copy.deepcopy(model.state_dict())
            print(f'Best model saved with accuracy: {best_acc:.2f}%')
        
        print(f'\nVal Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Best val acc: {best_acc:.2f}%, best acc at epoch = {best_epoch}')
        print('-' * 60)
        # print(f'\nValidation Confusion Matrix (Epoch {epoch}):')
        # for row in val_conf_matrix:
        #     # 6.3f means 6 total spaces, 3 after the decimal
        #     print(" ".join([f"{val:6.3f}" for val in row]))
        # print('-' * 80)

    print(f'Training completed!')
    print('-' * 80)
    

    best_model = CIFAR100_train(num_classes=num_classes).to(device)
    best_model.load_state_dict(best_model_state_dict)
    test_loss, test_acc, test_conf_matrix = test(best_model)
    print('\nTest Confusion Matrix:')
    for row in test_conf_matrix:
        # 6.3f means 6 total spaces, 3 after the decimal
        print(" ".join([f"{val:6.3f}" for val in row]))
    
    print('-' * 80)
    print(f'Best Validation Accuracy: {best_acc:.2f}%, at epoch: {best_epoch}, Test Accuracy: {test_acc:.2f}%')
    print('-' * 80)


    # ============================================================================
    # Saving everything
    # ============================================================================

    # Define the path for the results file
    results_path = 'CIFAR100_model.pth'

    # Create a dictionary containing all the data you want to preserve
    training_results = {
        'epoch_trained': epoch + 1,
        'learning_rate': learning_rate,
        'regularizer': regularizer,
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'best_val_acc': best_acc,
        'final_test_acc': test_acc,
        'final_test_conf_matrix': test_conf_matrix
    }

    # Save everything into one file
    torch.save(training_results, results_path)

    print(f"\n✓ All training results and best model saved to: {results_path}")

