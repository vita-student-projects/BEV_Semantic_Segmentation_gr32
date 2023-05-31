# DLAV Project: BEV Semantic Segmentation

### Group 32: Furkan Güzelant & Kardelen Ceren 
In this project, we worked on solving BEV Semantic Segmentation task using [PanopticBEV](https://github.com/robot-learning-freiburg/PanopticBEV) as our baseline model. 

## 1. Contribution Overview

### 1.1 Discarding Instance Segmentation 
![](https://hackmd.io/_uploads/rkRpUbEIn.png)

Based on the findings of the BEVformer model, which shows improved prediction accuracy by simultaneously performing 3D object detection and semantic segmentation, it is plausible that panoptic segmentation might exhibit similar behavior. As part of our research, we aim to compare these segmentation tasks and assess whether incorporating instance segmentation enhances the model's predictions.
### 1.2 Using a different backbone (ResNet)
[PanopticBEV](https://github.com/robot-learning-freiburg/PanopticBEV) uses EfficientDet-D3 model for its network backbone. As ResNet is widely employed as a feature extractor, another contribution we made is to utilize various ResNet variants as the network backbone. Specifically, we experimented with ResNet152 to assess the impact of different backbones on feature extraction within the network.
## 2. Experiments

### 2.1 Setup

We conducted our experiments on nuScenes dataset and ground truth data genereated by authors of PanopticBEV.
### 2.2 Comparing models

To conduct a comparative analysis, we trained two distinct models: one with the instance segmentation head and the other without it. Instead of the default 30 epochs, we trained the models for 6 epochs. In order to assess and compare the models, we employed semantic loss and semantic mIoU (mean Intersection over Union) as evaluation metrics.
 
### 2.3 Experimenting with backbones

We conducted an experiments on two different backbones, namely EfficientDet-D3 and ResNet152 to see their performance on the network. Similarly, we run each model for 6 epochs. We assessed their performance using semantic loss and semantic mIou.

### 2.4 Evaluation metric

We used semantic mIoU (mean Intersection over Union) to evaluate the performance of the models. 

![](https://hackmd.io/_uploads/HkvQnfNU3.png)
where Sp is the prediction and Sg is the ground truth


## 3. Dataset

### 3.1 Description of the dataset

The nuScenes dataset is specifically designed for a range of deep learning tasks related to autonomous driving. It comprises images and video sequences captured by six cameras positioned around the ego vehicle, in addition to radar and LIDAR inputs. The dataset consists of 1000 scenes, each lasting 20 seconds, resulting in approximately 1.4 million annotated images. The annotations are primarily focused on semantic segmentation and are generated using LIDAR data. The dataset encompasses 23 distinct classes, including pedestrians, various types of vehicles like trucks and bicycles, emergency vehicles, and different terrain types.

### 3.2 Label format & Dataset preparation

The ground truth data used in our study was created by the authors of PanopticBEV. You can find this dataset from   [here](http://panoptic-bev.cs.uni-freiburg.de/).

The nuScenes dataset can be found on the shared dataset in SCITAS at location:/work/scitas-share/datasets/Vita/civil-459/NuScenes_full/US

## 4. Setup

## 5. Results

## 6. Conclusion