# Parameter Efficient CNN Classifiers

This repository contains highly optimized CNN models focusing on high accuracy with a minimal parameter footprint. 




Approximately, ~35k and ~50k parameters for FashionMNIST, CIFAR10 datasets respectiv

## 🚀 Performance

Tiny models

| Model | Datasets | Parameters | Accuracy | Ensemble Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| FashionMNIST_35k_model1 | FashionMNIST | 35k | 92.7 % | **93.4 %** |
| FashionMNIST_35k_model2 | FashionMNIST | 35k | 92.5 % | |
| CIFAR10_56k_model1 | CIFAR10 | 56k | 85.5 % | **87.8 %** |
| CIFAR10_56k_model2 | CIFAR10 | 56k | 85.3 % | |

Slightly bigger models
| Model | Datasets | Parameters | Accuracy | Ensemble Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| CIFAR10_150k_model1 | CIFAR10 | 150k | 87.8 % |  |
| CIFAR10_150k_model2 | CIFAR10 | 150k | 87.3 % | |



## 🛠️ Usage
1. Clone the repo: `git clone https://github.com/MRayyanS/Parameter_efficient_CNNclassifiers.git`
2. Install dependencies: `pip install torch torchvision numpy matplotlib`
3. For training and reproduce: `dataset_train.py`
4. For inference: run `dataset_eval.py` with following changes (don't forget to select appropriate model/s)
    

