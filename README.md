# Parameter Efficient CNN Classifiers

This repository contains highly optimized CNN models focusing on high accuracy with a minimal parameter footprint. 




Approximately, ~35k and ~50k parameters for FashionMNIST, CIFAR10 datasets respectiv

## 🚀 Performance

Tiny models

| Model | Datasets | Parameters | Accuracy |
| :--- | :--- | :--- | :--- |
| FashionMNIST_35k_model1 | FashionMNIST | 35k | 92.7 % |
| FashionMNIST_35k_model2 | FashionMNIST | 35k | 92.5 % |
| CIFAR10_56k_model1 | CIFAR10 | 56k | 85.5 % |
| CIFAR10_56k_model2 | CIFAR10 | 56k | 85.3 % |

Slightly bigger models
| Model | Datasets | Parameters | Accuracy |
| :--- | :--- | :--- | :--- |
| CIFAR10_150k_model1 | CIFAR10 | 150k | 87.8 % |
| CIFAR10_150k_model2 | CIFAR10 | 150k | 87.3 % |


## Ensemble models



<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Datasets</th>
      <th>Parameters</th>
      <th>Accuracy</th>
      <th>Ensemble Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FashionMNIST_35k_model1</td>
      <td rowspan="2">FashionMNIST</td>
      <td>35k</td>
      <td>92.7 %</td>
      <td rowspan="2"><b>93.8 %</b></td>
    </tr>
    <tr>
      <td>FashionMNIST_35k_model2</td>
      <td>35k</td>
      <td>92.5 %</td>
    </tr>
    <tr>
      <td>CIFAR10_56k_model1</td>
      <td rowspan="2">CIFAR10</td>
      <td>56k</td>
      <td>85.5 %</td>
      <td rowspan="2"><b>87.2 %</b></td>
    </tr>
    <tr>
      <td>CIFAR10_56k_model2</td>
      <td>56k</td>
      <td>85.3 %</td>
    </tr>
  </tbody>
</table>


## 🛠️ Usage
1. Clone the repo: `git clone https://github.com/MRayyanS/Parameter_efficient_CNNclassifiers.git`
2. Install dependencies: `pip install torch torchvision numpy matplotlib`
3. For training and reproduce: `python train_procedure.py`
4. For inference: run `ensamble_models.py` with following changes
    a. select appropriate dataset_name = "FashionMNIST", "CIFAR10" (line 10)
    b. select appropriate collection of models (line 51)
    c. select appropriate model class (line 58), the definitions of model class are provided in `model_architechtures.py`


