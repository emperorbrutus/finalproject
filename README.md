# Heart MRI Segmentation Project

This project focuses on segmenting different parts of the heart using deep learning techniques. We utilize DenseNet-121 and MobileNetV2 models to accurately segment regions of interest in heart MRI images. The dataset used for training and evaluation is from the Automated Cardiac Diagnosis Challenge (ACDC) at MICCAI 2017.

## Table of Contents

- [Introduction](#introduction)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Weights](#weights)
- [Links](#links)

## Introduction

This project aims to develop an image segmentation model for heart MRI images. The primary goal is to accurately segment the left ventricle (LV), myocardium (MYO), and right ventricle (RV).

## Objectives

- Segment the left ventricle (LV)
- Segment the myocardium (MYO)
- Segment the right ventricle (RV)

## Dataset

The dataset used in this project is from the Automated Cardiac Diagnosis Challenge (ACDC) at MICCAI 2017. It includes annotated MRI images of the heart. The dataset can be accessed through the link [here](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb)

## Models

### DenseNet-121

DenseNet-121 is a densely connected convolutional network architecture that improves the flow of information and gradients throughout the network, making it easier to train. It is known for achieving high performance on image classification tasks. The pre-trained model can be access through the link [here](https://download.pytorch.org/models/densenet121-a639ec97.pth)

### MobileNetV2

MobileNetV2 is a lightweight convolutional neural network designed for mobile and edge devices. Despite its smaller size and lower computational cost, it can achieve competitive accuracy. This model is less explored for the given dataset, hence its inclusion in this project aims to evaluate its performance in heart MRI segmentation. The pre-trained model can be access through the link [here](https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth)

## Results

The models were evaluated using class-specific Dice scores, which measure the overlap between the predicted and ground truth segmentations. The results are summarized below:

### MobileNetV2 Dice Coefficient Scores

| Dataset      | Right Ventricle (RV) | Myocardium (MYO) | Left Ventricle (LV) |
|--------------|----------------------|------------------|---------------------|
| Training Set | 0.44675              | 0.78153          | 0.84229             |
| Testing Set  | 0.43234              | 0.74065          | 0.82345             |

### DenseNet-121 Dice Coefficient Scores

| Dataset      | Right Ventricle (RV) | Myocardium (MYO) | Left Ventricle (LV) |
|--------------|----------------------|------------------|---------------------|
| Training Set | 0.7178               | 0.8388           | 0.8764              |
| Testing Set  | 0.6708               | 0.8096           | 0.8429              |

## Links
The image results, model weights, and additional training data can be accessed through the link [here](https://drive.google.com/drive/folders/1hKIaiz7B5Ge-yhUTF9ChiaOTxnQyB8px?usp=sharing). This includes detailed visualizations of the segmentation outputs for the Heart MRI dataset, showcasing the performance of both DenseNet-121 and MobileNetV2 models. By examining these results, one can gain insights into the efficacy of the models in accurately identifying and segmenting the Left Ventricle, Myocardium, and Ventricle regions. The provided link also contains the model weights, allowing for replication of the experiments or further fine-tuning to improve segmentation accuracy.
