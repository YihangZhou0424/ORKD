# Online Relational Knowledge Distillation for Image Classification

This project provides source code for official implementation of  Online Relational Knowledge Distillation (ORKD):

## Installation

### Requirements

Ubuntu 18.04 LTS

Python 3.8

CUDA 11.1

Create three folders `./data`, `./result`, and `./checkpoint`,

please install python packages:

```
pip install -r requirements.txt
```

## Perform experiments on CIFAR-100 dataset

### Dataset

CIFAR-100 : [download](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

unzip to the `./data` folder

### Training for baseline
Training DenseNet-40-12, ResNet-32, VGG-16, ResNet-110, HCGNet-A1 baseline:
```
python main_cifar_baseline.py --arch densenetd40k12
python main_cifar_baseline.py --arch resnet32
python main_cifar_baseline.py --arch vgg16
python main_cifar_baseline.py --arch resnet110
python main_cifar_baseline.py --arch hcgnet_A1
```

### Training by ORKD
Training DenseNet-40-12, ResNet-32, VGG-16, ResNet-110, HCGNet-A1 with 4 peer networks:
```
python main_cifar_orkd.py --arch mcl_okd_densenetd40k12
python main_cifar_orkd.py --arch mcl_okd_resnet32
python main_cifar_orkd.py --arch mcl_okd_vgg16
python main_cifar_orkd.py --arch mcl_okd_resnet110
python main_cifar_orkd.py --arch mcl_okd_hcgnet_A1
```

