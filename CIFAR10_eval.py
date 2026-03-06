import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from model_architectures import *
from train_procedure import *

import warnings
warnings.filterwarnings("ignore", message=".*VisibleDeprecationWarning.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Global setup
# ============================================================================
np.random.seed(750)
torch.manual_seed(750)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

DATA_mean = (0.4914, 0.4822, 0.4465)
DATA_std  = (0.2023, 0.1994, 0.2010)

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(DATA_mean, DATA_std),
])

test_data = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=eval_transform
)

test_loader = DataLoader(
    test_data, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=4,
)


# ============================================================================
# Ensemble: load 5 models
# ============================================================================

MODEL_PATHS = [
    f"trained_models/CIFAR10_150k_training_results{i}.pth" for i in [1, 2]
]

def load_ensemble(model_paths, device):
    """Load all models and return them in eval mode."""
    models = []
    for path in model_paths:
        m = CIFAR10_150k(num_classes=num_classes).to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(checkpoint['model_state_dict'])
        m.eval()
        models.append(m)
        print(f"Loaded model from '{path}'  "
              f"(best val acc: {checkpoint.get('best_val_acc', 'N/A'):.2f}%, "
              f"test acc: {checkpoint.get('final_test_acc', 'N/A'):.2f}%)")
    return models


def ensemble_predict(models, images):
    """
    Given a batch of images, sum the raw logits from all models,
    then apply softmax to obtain class probabilities.

    Returns:
        probs      (B, C) – class probabilities
        predicted  (B,)   – predicted class indices
    """
    with torch.no_grad():
        logit_sum = 0
        for m in models:
            logits = m(images)                    # (B, C) raw scores
            logit_sum += logits

        probs     = F.softmax(logit_sum, dim=1)   # sum-of-logits → softmax
        predicted = probs.argmax(dim=1)
    return logit_sum, probs, predicted


# ============================================================================
# Test function (mirrors the original test() from new_FCN.py)
# ============================================================================

def test_ensemble(models, test_loader, criterion):
    print("Testing ensemble on the test set …")

    correct   = 0
    total     = 0
    test_loss = 0.0

    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.float32)
    class_counts     = torch.zeros(num_classes,              dtype=torch.float32)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            logit_sum, probs, predicted = ensemble_predict(models, images)

            # Compute loss on summed logits
            loss = criterion(logit_sum, labels)

            test_loss += loss.item()
            total     += labels.size(0)
            correct   += predicted.eq(labels).sum().item()

            for t in range(len(labels)):
                true_class = labels[t].long()
                confusion_matrix[true_class] += probs[t].cpu()
                class_counts[true_class]     += 1

    accuracy  = 100.0 * correct / total
    avg_loss  = test_loss / len(test_loader)

    confusion_matrix_avg = confusion_matrix / class_counts.unsqueeze(1)

    print(f"\nEnsemble Test Accuracy : {accuracy:.2f}%")
    print(f"Ensemble Test Loss     : {avg_loss:.4f}")
    print("\nTest Confusion Matrix (row = true class, col = predicted class):")
    for row in confusion_matrix_avg.numpy():
        print(" ".join([f"{v:6.3f}" for v in row]))

    return avg_loss, accuracy, confusion_matrix_avg.numpy()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()

    # Load all 5 models
    models = load_ensemble(MODEL_PATHS, device)
    print(f"\nEnsemble of {len(models)} models ready on device: {device}\n")

    # Evaluate on test set
    test_loss, test_acc, test_conf_matrix = test_ensemble(models, test_loader, criterion)