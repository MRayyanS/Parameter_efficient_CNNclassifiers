# Parameter Efficient CNN Classifiers

This repository contains highly optimized CNN models focusing on high accuracy with a minimal parameter footprint.

## 🚀 Performance

Tiny models

| Model | Datasets | Parameters | Accuracy | Ensemble Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| FashionMNIST_35k_model1 | FashionMNIST | 35k | 92.70 % | **93.41 %** |
| FashionMNIST_35k_model2 | FashionMNIST | 35k | 92.55 % | |
| CIFAR10_56k_model1 | CIFAR10 | 56k | 85.54 % | **87.78 %** |
| CIFAR10_56k_model2 | CIFAR10 | 56k | 85.33 % | |



Slightly bigger models

| Model | Datasets | Parameters | Accuracy | Ensemble Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| FashionMNIST_90k_model1 | FashionMNIST | 90k | 93.19 % | **94.11 %** |
| FashionMNIST_90k_model2 | FashionMNIST | 90k | 93.08 % | |
| CIFAR10_150k_model1 | CIFAR10 | 150k | 87.80 % | **89.97 %** |
| CIFAR10_150k_model2 | CIFAR10 | 150k | 87.33 % | |
| CIFAR100_277k_model1 | CIFAR100 | 277k | 61.19 % | **65.13 %** |
| CIFAR100_277k_model2 | CIFAR100 | 277k | 61.60 % | |



# Design and training philosophy 


High parameter efficiency while maintaining decent performance is achieved by the following design and training philosophy:

### 1. ***Bottleneck Feature Extraction***
The architecture utilizes a "squeeze-and-expand" strategy, employing convolutional bottleneck layers (ExpComp module) to extract features with high kernel variety. This approach allows the model to capture a wide array of initial features while consistently reducing large channel counts (e.g., 128) down to very small values (2 or 4) immediately in the initial layers.

### 2. ***Depthwise Separable Layers for Residual Feature Refinement*** 
In the deeper layers, where feature refinement occurs through residual connections, parameter efficiency is achieved by combining bottleneck structures with depthwise separable convolutions. This decoupling of spatial and channel-wise learning maintains representational power without the massive parameter cost of standard convolutional stacks.

### 3. ***Fully Convolutional Heads*** 
To eliminate the high parameter footprint of traditional Multi-Layer Perceptrons (MLPs), the final classification heads are designed to be fully convolutional. The model transitions directly from feature refinement to prediction using 3x3 convolutions and global pooling, keeping the end-to-end architecture lean and efficient.

### 4. ***Robust Training and Multi-Stage Augmentation*** 
The training process incorporates a multi-stage augmentation strategy that combines standard spatial transformations—such as random horizontal flips and rotations—with advanced regularization techniques like Random Erasing and custom dynamic Gaussian noise and blur that scale in intensity throughout the training epochs.


## 🛠️ Usage

1. For training and reproduce: `[dataset]_train.py`
2. For inference: run `[dataset]_eval.py`, (don't forget to select appropriate model/s)
3. "trained_models" directory contains all the trained models (open weights)
4. "model_architecture.py" contains the details of the model architecture for all the trained models
    

