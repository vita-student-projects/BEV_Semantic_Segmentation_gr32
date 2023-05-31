# DLAV Project: BEV Semantic Segmentation

### Group 32: Furkan Güzelant & Kardelen Ceren 
In this project, we worked on solving BEV Semantic Segmentation task using [PanopticBEV](https://arxiv.org/abs/2108.03227) as our baseline model. Repository forked from [here](https://github.com/robot-learning-freiburg/PanopticBEV).

## 1. Contribution Overview

### 1.1 Discarding Instance Segmentation 
![](https://hackmd.io/_uploads/ByUgECEIh.png)

Based on the findings of the BEVformer model, which shows improved prediction accuracy by simultaneously performing 3D object detection and semantic segmentation, it is plausible that panoptic segmentation might exhibit similar behavior. Since PanopticBEV has not reported ablation studies done on this subject, as part of our research, we aim to compare these segmentation tasks and assess whether incorporating instance segmentation enhances the model's predictions.

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

The nuScenes dataset can be found on the shared dataset in SCITAS at location:  /work/scitas-share/datasets/Vita/civil-459/NuScenes_full/US

## 4. Setup

To run this model in SCITAS, go through the following steps, while connected to an interactive session.

Setup the environment and the model:

```
module load gcc/8.4.0-cuda python/3.7.7 cuda/11.1.1
python3 -m venv panoptic_bev
source panoptic_bev/bin/activate

pip install --user torch==1.8.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install --user torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip3 install --user -r requirements.txt

python setup.py develop --user
```
Download the PanopticBEV dataset mentioned above from [here](http://panoptic-bev.cs.uni-freiburg.de/).

### Training and Evaluation
In the "scripts" folder, model training is done in "train_panoptic_bev.py" and evaluation is done in "eval_panoptic_bev.py", which are called by scripts "train_panoptic_bev_nuscenes.sh" and "eval_panoptic_bev_nuscenes.sh", respectively. Modify these last two files so that: 

* project_root_dir: the root of PanopticBEV in the server
* seam_root_dir: location of the downloaded PanopticBEV dataset
* run_name: a folder with this name is created each time the script is run, so must be changed from run to run
* resume (only in eval): location of the saved model to evaluate. Can be found in "experiments/run_name/saved_models/model_best.pth"

Run training and evaluation scripts. Config can be found in "experiments/config/nuscenes.ini". Logs will be written in "experiments/run_name/logs".

### Pre-trained models
You can access the weights [here.](https://drive.google.com/drive/folders/1xoWRH4V4Y2Zmw80EB948CmT-rZsE9sIh?usp=sharing)

## 5. Results
Semantic segmentation results by class, in IoU [%], sorted by mIoU: 
| Model|  Instance Seg. |Epochs | Road  | Side. | Manm. | Veg. | Ter. | Occ. | Per. | 2-Wh. | Car  | Truck | mIoU  |
| ------------ | ------ | --- | ----- | -------- | ------- | ---------- | ------- | --------- | ------ | --------- | ----- | ----- | ----- |
| PanopticBEV  | Yes |  6 | 65.84 | 19.84    | 31.54  | 28.77 | 27.01| 28.81 | 2.89| 3.58 | 28.60 | 25.41 | 26.23   |
| PanopticBEV |   No   | 6 | 67.47 | 18.60    | 29.76   | 30.75 | 27.13| 30.51 | 3.60  | 5.20 | 29.85 | 25.70 | 26.85 |
| PanopticBEV  with Resnet| Yes |  6 | 73.25 | 25.01    | 36.25  | 34.90 | 33.10 | 33.55 | 2.66 | 5.37 | 35.67 | 25.48 | 30.52 |
| PanopticBEV |   No   | 10 | 76.49 | 28.05 | 36.25  | 34.44    | 34.24  | 34.23 | 4.85 | 7.50 | 38.83| 30.04 | 32.49 |
| PanopticBEV† |   No   | 30 | 77.32 | 28.55    | 36.72   | 35.06      | 33.56   | 36.65     | 4.98   | 9.63      | 40.53 | 33.47 | 33.65 |


†: as reported in the [PanopticBEV paper](https://arxiv.org/pdf/2108.03227.pdf).

It can be seen that training the original model with 10 epochs, we attained comparable results to those reported in the paper. However, due to time constraints, we performed our experiments with 6 epochs, and chose our baseline as the original model with 6 epochs. 

We observe that training the model with instance segmentation head reduces the semantic segmentation performance. However, using the adapted ResNet as the model's backbone decidedly increases model performance. 

![](https://hackmd.io/_uploads/S1YP44rI2.png)

As can be seen in the above graph, in all the experiments, the loss continues to decrease. We expect that if we were to run the same experiments with 30 epochs as the paper suggested, we would get similar or better results. 

An example BEV semantic segmentation output from our no instance segmentation, 6 epochs experiment: 
![](https://hackmd.io/_uploads/SkkI0eBUn.png)
You can see that even with the occlusion (rain drops), the model was able to identify the road, and the cars ahead. 

## 6. Conclusion

In this project, we modify the PanopticBEV model to understand the effects of training both semantic segmentation and instance segmentation tasks at the same time and the backbone choice. We conclude that training with both task heads slightly decreases the semantic segmentation performance and that ResNet as a backbone increases the performance. 