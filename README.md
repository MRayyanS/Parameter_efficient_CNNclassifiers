# Parameter Efficient Classifiers

This repository contains highly optimized models focusing on high accuracy with a minimal parameter footprint.

## 🚀 Performance

Tiny models

| Model | Datasets | Parameters | Accuracy | Ensemble Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| fMNIST_35k_model1 | FashionMNIST | 35k | 92.70 % | **93.41 %** |
| fMNIST_35k_model2 | FashionMNIST | 35k | 92.55 % | |
| CIFAR10_conv_45k_model1 | CIFAR10 | 45k | 86.05 % | **88 %** |
| CIFAR10_conv_45k_model2 | CIFAR10 | 45k | 85.09 % | |



Mid-size models

| Model | Datasets | Parameters | Accuracy | 
| :--- | :--- | :--- | :--- |
| fMNIST_90k_model1 | FashionMNIST | 90k | 93.19 % |
| fMNIST_90k_model2 | FashionMNIST | 90k | 93.08 % |
| CIFAR10_conv_78k_model1 | CIFAR10 | 78k | 87.13 % |
| CIFAR10_trans_88k_model1 | CIFAR10 | 88k | 85.81 % |



Slightly bigger models

| Model | Datasets | Parameters | Accuracy | 
| :--- | :--- | :--- | :--- |
| CIFAR10_conv_207k_model1 | CIFAR10 | 207k | 89.71 % |
| CIFAR100_326k_model1 | CIFAR100 | 326k | 62.95 % |
| CIFAR100_277k_model1 | CIFAR100 | 277k | 61.19 % | 
| CIFAR100_277k_model2 | CIFAR100 | 277k | 61.60 % |



# Design and training philosophy 


High parameter efficiency while maintaining decent performance is achieved by the following design and training philosophy:

### 1. ***Bottleneck Feature Extraction***
The architecture utilizes a "squeeze-and-expand" strategy, employing convolutional bottleneck layers (ExpComp module) to extract features with high kernel variety. This approach allows the model to capture a wide array of initial features while consistently reducing large channel counts (e.g., 128) down to very small values (2 or 4) immediately in the initial layers.

### 2. ***Depthwise Separable Layers for Residual Feature Refinement*** 
In the deeper layers, where feature refinement occurs through residual connections, parameter efficiency is achieved by combining bottleneck structures with depthwise separable convolutions. This decoupling of spatial and channel-wise learning maintains representational power without the massive parameter cost of standard convolutional stacks.

### 3. ***Fully Convolutional Heads*** 
To eliminate the high parameter footprint of traditional Multi-Layer Perceptrons (MLPs) towards the end of the classifiers, the final classification heads are designed to be fully convolutional. The model transitions directly from feature refinement to prediction using 3x3 convolutions at first, and then two layers of 1x1 kernels imitating an MLP for each spatial feature vecture. Finally global pooling leads to taking an ensemble of all the spacial features resulting in better generalization, keeping the end-to-end architecture lean and efficient.

### 4. ***Robust Training and Multi-Stage Augmentation*** 
The training process incorporates a multi-stage augmentation strategy that combines standard spatial transformations—such as random horizontal flips and rotations—with advanced regularization techniques like Random Erasing and custom dynamic Gaussian noise and blur that scale in intensity throughout the training epochs.


## 🛠️ Usage

1. For training and reproduce: `[dataset]_train.py`
2. For inference: run `[dataset]_eval.py`, (don't forget to select appropriate model/s)
3. "trained_models" directory contains all the trained models (open weights)
4. "model_architecture.py" contains the architecture details of all the trained models
    

