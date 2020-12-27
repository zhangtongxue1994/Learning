# cv2 detect and localization
#coding=utf-8 
import cv2
import numpy as np
from matplotlib import pylab as plt
import argparse
import glob
from math import pi



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN tutorial.')
    parser.add_argument('--input1', help='Path to input image 1.', default='rec_small1.jpg')
    parser.add_argument('--input2', help='Path to input image 2.', default='frame.jpg')
    args = parser.parse_args()
    img_object = cv2.imread(args.input1, cv2.IMREAD_GRAYSCALE)
    img_scene = cv2.imread(args.input2, cv2.IMREAD_GRAYSCALE)
    if img_object is None or img_scene is None:
        print('Could not open or find the images!')
        exit(0)
    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    minHessian = 400
    # detector = cv2.xfeatures2d. SURF_create()(hessianThreshold=minHessian)
    detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
    keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)
    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)
    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    #-- Draw matches
    img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #-- Localize the object
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
        #-- Get the keypoints from the good matches
        obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
    H, _ =  cv2.findHomography(obj, scene, cv2.RANSAC)
    #-- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((4,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0
    obj_corners[0,0,1] = 0
    obj_corners[1,0,0] = img_object.shape[1]
    obj_corners[1,0,1] = 0
    obj_corners[2,0,0] = img_object.shape[1]
    obj_corners[2,0,1] = img_object.shape[0]
    obj_corners[3,0,0] = 0
    obj_corners[3,0,1] = img_object.shape[0]
    scene_corners = cv2.perspectiveTransform(obj_corners, H)
    #-- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv2.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
        (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
    cv2.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
        (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
    cv2.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
        (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
    cv2.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
        (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)
    cv2.circle(img_matches,(int(img_object.shape[0]),int(img_object.shape[1])),3,(0,255,0))
    #-- Show detected matches
    cv2.imwrite('Good Matches&Object detection1.jpg',img_matches)
    cv2.namedWindow('Good Matches & Object detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Good Matches & Object detection", 640, 480);
    cv2.imshow('Good Matches & Object detection', img_matches)
    cv2.waitKey(0)
    # shape[0]代表高度，shape[1]代表宽度
    # print(img_object.shape[0],img_object.shape[1])

    #输出对应顺序为CDAB
    ABCD = ([int(scene_corners[2,0,0]),int(scene_corners[2,0,1])],
            [int(scene_corners[3,0,0]),int(scene_corners[3,0,1])],
            [int(scene_corners[0,0,0]),int(scene_corners[0,0,1])],
            [int(scene_corners[1,0,0]),int(scene_corners[1,0,1])])






