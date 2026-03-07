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
| CIFAR10_150k_model1 | CIFAR10 | 150k | 87.80 % | **89.97 %** |
| CIFAR10_150k_model2 | CIFAR10 | 150k | 87.33 % | |





## 🛠️ Usage

1. For training and reproduce: `[dataset]_train.py`
2. For inference: run `[dataset]_eval.py` with following changes (don't forget to select appropriate model/s)
3. "trained_models" directory contains all the trained models (open weights)
4. "model_architecture.py" contains the details of the model architecture for all the trained models
    

