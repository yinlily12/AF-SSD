# coding: utf-8

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
import colorsys
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from AF_SSD import build_net
import matplotlib
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
from data import VOC_CLASSES as labels

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

net = build_net('test', 300, 11)    # initialize AF_SSD
net.load_weights('../weights/AFSSD_VOC_60000.pth')
net.eval()

matplotlib.use('TkAgg')
def vis_detections(im, class_name, dets,color):
    bbox = tuple(int(np.round(x)) for x in dets[:4])
    cv2.rectangle(im, bbox[0:2], bbox[2:4], color, 2)
    cv2.putText(im, (class_name), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN,
                2.0, (0,0,255), thickness=2)
    return im
imagedir='../demo/'
savedir='../demo/'

imagename='100.jpg'
# import pdb
# pdb.set_trace()
image = cv2.imread(imagedir+imagename)
rgb_image =np.copy(image)
imshow =np.copy(image)

# fig=plt.figure(figsize=(10,10))
fig=plt.figure()
img_input = cv2.resize(image, (300, 300)).astype(np.float32)

img_input -= (104.0, 117.0, 123.0)
img_input = img_input.astype(np.float32)
img_input = img_input[:, :, ::-1].copy()
img_input = torch.from_numpy(img_input).permute(2, 0, 1)

var_img_input = Variable(img_input.unsqueeze(0))     # wrap tensor in Variable
if torch.cuda.is_available():
    var_img_input = var_img_input.cuda()
y = net(var_img_input)

colors = plt.cm.hsv(np.linspace(0, 1, 11)).tolist()
plt.imshow(rgb_image)  # plot the image for matplotlib
currentAxis = plt.gca()

detections = y.data

scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
for i in range(detections.size(1)):
    j=0
    #show boxes with score>0.5
    while j<detections.size(2) and detections[0,i,j,0] >= 0.5:
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        coords = np.array([pt[0], pt[1], pt[2], pt[3]])
        color = colors[i]
        color_new=colorsys.hsv_to_rgb(color[0],color[1],color[-1])
        color1=[]
        for k in range(len(color_new)):
            color1.append(int(color_new[k]*256))

        imshow=vis_detections(imshow, label_name, coords,color1)
        j+=1
# plt.show()
savepath = savedir+imagename.split('.')[0]+'_det.jpg'
cv2.imwrite(savepath,imshow)
