"""
------------------------------------------------
File Name: predict.py
Author: zhangtongxue
Date: 2019/11/21 22:22
Description:
Reference:https://github.com/idealo/image-super-resolution
-------------------------------------------------
"""
import numpy as np
from PIL import Image
from ISR.models import RDN, RRDN
import cv2

img = Image.open('baboon.png')
lr_img = np.array(img)

# RDN predict
rdn = RDN(arch_params={'C': 6, 'D': 20, 'G': 64, 'G0': 64, 'x': 2})
# rdn.model.load_weights('weights/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')
rdn.model.load_weights('weights/rdn-C6-D20-G64-G064-x2/2019-11-22-17_25/rdn-C6-D20-G64-G064-x2_best-val_loss_epoch002.hdf5')
sr_img = rdn.predict(lr_img)
# Image.fromarray(sr_img)

# # RRDN predict
# scale = 2
# rrdn = RRDN(arch_params={'C': 4, 'D': 3, 'G': 64, 'G0': 64, 'T': 10, 'x': scale})
# rrdn.model.load_weights(
#     'weights/rrdn-C4-D3-G64-G064-T10-x2/2019-11-22-16_57/rrdn-C4-D3-G64-G064-T10-x2_best-val_loss_epoch001.hdf5')
# sr_img = rrdn.predict(lr_img)
# # sr_img = rrdn.predict(lr_img, by_patch_of_size=50)

cv2.imshow('sr_img', sr_img)
cv2.waitKey(0)
