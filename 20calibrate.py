#!/usr/bin/python2

import numpy as np
import cv2
import os, sys
import PIL.Image

if __name__ == '__main__':
    # ./this.py chess_pics/small/*
    img_names = sys.argv[1:]

    # maxiter=1000 and tol=0.0001 makes results worse again (too much
    # counter-bending at the image corners)
    maxiter = 1000 # 30
    tol = 0.1 # 0.1
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, maxiter, tol)
    
    # original images scaled down by factor `fraction`
    fraction = 0.2

    scale_h = scale_w = img_points_scale = 1.0/fraction
    # exact scale factors based on actual image sizes, might differ very
    # slightly from 1/fraction and from one another
##    scale_w = 3264/653.0
##    scale_h = 2448/490.0
##    img_points_scale = np.array([scale_w, scale_h], dtype=np.float32)[None,:]
    
    # size of chessboard pattern (number of corners)
    pattern_size = (9, 6)
    # rectangular grid of chessboard corners, viewed along chessboard plane
    # normal vector; z coord is zero; the "real" object = the undistorted chessboard
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32 )
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points_lst = [pattern_points]*len(img_names)

    img_points = []
    h_old, w_old = None, None
    for fn in img_names:
        print("processing {}".format(fn))
        # XXX rotate -- or not???
        # http://sylvana.net/jpegcrop/exif_orientation.html
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
        # http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#void%20initUndistortRectifyMap%28InputArray%20cameraMatrix,%20InputArray%20distCoeffs,%20InputArray%20R,%20InputArray%20newCameraMatrix,%20Size%20size,%20int%20m1type,%20OutputArray%20map1,%20OutputArray%20map2%29
        pil_img = PIL.Image.open(fn)
        print(pil_img._getexif()[274])

        # convert grayscale
        img = np.array(pil_img.convert('L'))
##        img = cv2.imread(fn, 0)
        h, w = img.shape[:2]

        if h_old:
            assert h == h_old, "h={}, h_old={}".format(h, h_old)
            assert w == w_old
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            print("refine")
            # corners_fine: scale up here and refine on scaled coords,
            # should gain some small percent in precision, but not much
            corners_fine = corners.copy() * img_points_scale
            cv2.cornerSubPix(img, corners_fine, (5, 5), (-1, -1), term)
            cv2.drawChessboardCorners(img, pattern_size, corners, found)
            cv2.imshow('img',img)
            cv2.waitKey(5000)
            img_points.append(corners_fine.reshape(-1, 2))
            h_old, w_old = h, w
            print('ok')
        else:
            print('chessboard not found')
            continue
        
   
    print("calibrateCamera")
    # int stuff only when scaling, could also simply use the orig w and h of
    # the orig images :)
    size = (int(round(w*scale_w)), int(round(h*scale_h)))
    rms, camera_matrix, coeffs, rvecs, tvecs = cv2.calibrateCamera(pattern_points_lst,
                                                                   img_points, 
                                                                   size,
                                                                   None,
                                                                   None, None,
                                                                   None,
                                                                   cv2.CALIB_RATIONAL_MODEL,
                                                                   term)
    # XXX not sure about the math here
    mean_error = 0
    for i in range(len(pattern_points_lst)):
	imgpoints2, _ = cv2.projectPoints(pattern_points_lst[i], rvecs[i], tvecs[i], camera_matrix, coeffs)
	error = cv2.norm(img_points[i],imgpoints2.reshape(-1,2), cv2.NORM_L2)/len(imgpoints2)
	mean_error += error

    print "mean error: ", mean_error/len(pattern_points_lst)
    
    print "RMS:", rms
    print "camera matrix:\n", camera_matrix
    print "distortion coefficients: ", coeffs.ravel()
    np.save('camera_matrix.npy', camera_matrix)
    np.save('coeffs.npy', coeffs)
    # XXX only if we use drawChessboardCorners() !!!
    cv2.destroyAllWindows()
