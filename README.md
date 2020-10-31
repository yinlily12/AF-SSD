# AF-SSD: An Accurate and Fast Single Shot Detector for High Spatial Remote Sensing Imagery
by Ruihong Yin, Wei Zhao, Xudong Fan, Yongfeng Yin
## Introduction


## Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training'>Training</a>
- <a href='#evaluation'>Evaluation</a>
- <a href='#demo'>Demo</a>
- <a href='#performance'>Performance</a>


## Installation
* Install PyTorch 0.4.0 by the instrument on the website [Pytorch](https://pytorch.org/) and running the approriate command.
* Clone this repository. This repository is mainly based on [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch), a huge thank to them.
  - *`Note`*: We currently only support Python 3+ and Pytorch 0.4.0.
  
## Datasets
NWPU VHR-10 dataset is avalable here.

## Training 
* Pre-trained MobileNetv1 is downloaded [here](https://pan.baidu.com/s/1SLdpsta035SxnxWfe-09UA)(code :`h4y7`). 
## Evaluation

## Demo

## Performance
### NWPU VHR-10
|System | mAP |Average Running Time|
|:--:|:--:|:--:|
|COPD|54.6%|1.070s|
|YOLOv2|60.5%|**0.026s**|
|RICNN|72.6%|8.770s|
|R-P-Faster RCNN|76.5%|0.150s|
|NEOON|77.5%|0.059s|
|SSD*|80.5%|0.042s|
|Faster RCNN|80.9%|0.430s|
|CACMOD CNN|**90.4**%|2.700s|
|AF-SSD|**88.7**%|**0.035s**|

*`Note`*: 
 - The result of SSD `SSD*` is our reproduced result with the same parameters as AF-SSD.
 - The testing environment is NVIDIA GTX-1080Ti.
