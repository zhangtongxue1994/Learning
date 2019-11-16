"""
------------------------------------------------
File Name: FasterRCNN.py
Author: zhangtongxue
Date: 2019/11/2 11:19
Description:
Reference: https://pytorch.org/docs/stable/torchvision/models.html#\
object-detection-instance-segmentation-and-person-keypoint-detection
-------------------------------------------------
"""

import os
import numpy as np
import torch
from PIL import Image
import cv2

import torchvision

from torchsummary import summary

if __name__ == "__main__":
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    x = [torch.from_numpy(cv2.imread('./PennFudanPed/PNGImages/FudanPed00001.png')).float().permute(2, 0, 1)]
    predictions = model(x)
    print(predictions)

    # summary(model, input_size=(224, 224))
    print(model)