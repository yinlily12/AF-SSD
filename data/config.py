#coding=utf-8
import os.path

# gets home dir cross platform
# HOME = os.path.expanduser("~")
HOME = '.'#os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

voc = {
    'num_classes': 11,
    'lr_steps': (500,30000, 40000,50000),
    'max_iter': 60000,
    'feature_maps': [75,38, 19, 10],
    'min_dim': 300, #image size
    'steps': [4,8, 16, 32],
    'min_sizes': [15, 30, 60, 111],
    'max_sizes': [30, 60, 111, 315],
    'aspect_ratios': [[2],[2], [2,3], [2,3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
