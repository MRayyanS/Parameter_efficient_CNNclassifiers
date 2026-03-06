# Parameter Efficient CNN Classifiers

This repository contains highly optimized CNN models for FashionMNIST, focusing on high accuracy with a minimal parameter footprint (~35k and ~50k parameters).

## 🚀 Performance
| Model | Parameters | Accuracy |
| :--- | :--- | :--- | :--- |
| Model V1 | 35k | [Your %] |
| Model V2 | 50k | [Your %] |

## 🛠️ Usage
1. Clone the repo: `git clone https://github.com/MRayyanS/Parameter_efficient_CNNclassifiers.git`
2. Install dependencies: `pip install torch torchvision numpy matplotlib`
3. For training and reproduce: `python train_procedure.py`
4. For inference: run `ensamble_models.py` with following changes
    a. select appropriate dataset_name = "FashionMNIST", "CIFAR10" (line 10)
    b. select appropriate collection of models (line 51)
    c. select appropriate model class (line 58), the definitions of model class are provided in `model_architechtures.py`


