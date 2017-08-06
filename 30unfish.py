#!/usr/bin/python2

import os, sys
import numpy as np
import cv2
pj = os.path.join

# https://hackaday.io/project/12384-autofan-automated-control-of-air-flow/log/41862-correcting-for-lens-distortions
# http://docs.opencv.org/2.4
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html

camera_matrix = np.load('camera_matrix.npy')
coeffs = np.load('coeffs.npy')
dr = 'converted'


if not os.path.exists(dr):
    os.makedirs(dr)


# ./this.py real_pics/orig/*
for ifn,fn in enumerate(sys.argv[1:]):
    print(fn)
    src = cv2.imread(fn)
    hh,ww = src.shape[:2]
    if ifn == 0:
        size_src = (ww, hh)
        camera_matrix_new, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                       coeffs, size_src, 0,
                                                       size_src)
    assert (ww,hh) == size_src
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, coeffs, 
                                             None, camera_matrix_new, size_src,
                                             cv2.CV_32FC1)
    
    dst = cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR)
    
    cv2.imwrite(pj(dr, os.path.basename(fn)), dst)
