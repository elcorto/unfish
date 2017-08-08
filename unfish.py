#!/usr/bin/python2

import numpy as np
import cv2, json
import os, sys, argparse
import PIL.Image
pj = os.path.join

# https://hackaday.io/project/12384-autofan-automated-control-of-air-flow/log/41862-correcting-for-lens-distortions
# http://docs.opencv.org/2.4
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
# https://codeyarns.com/2014/01/16/how-to-convert-between-numpy-array-and-pil-image/


def calibrate(img_names, fraction=0.2, maxiter=30, tol=0.1, pattern_size=(9,6)):
    # pattern_size = size of chessboard pattern (number of corners)
    # maxiter, tol
    # termination criteria for calibrateCamera
    #
    # maxiter=1000 and tol=0.0001 makes results worse again (too much
    # counter-bending at the image corners)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, maxiter, tol)
    
    # fraction:
    # original images scaled down by factor `fraction`: chess_pics/small used
    # for calibration instead of chess_pics/orig b/c it is MUCH faster and the
    # accuracy of findChessboardCorners on the scaled down calibration images
    # is by far enough
    scale_h = scale_w = img_points_scale = 1.0/fraction
    # instead of fraction, here are the
    # exact scale factors based on actual image sizes, might differ very
    # slightly from 1/fraction and from one another
##    scale_w = 3264/653.0
##    scale_h = 2448/490.0
##    img_points_scale = np.array([scale_w, scale_h], dtype=np.float32)[None,:]
   
    # XXX quite involved numpyology here, can this be some in a simpler way??
    # rectangular grid of chessboard corners, viewed along chessboard plane
    # normal vector; z coord is zero; the "real" object = the undistorted chessboard
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32 )
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    
    pattern_points_lst = []
    img_points_lst = []
    h_old, w_old = None, None
    for fn in img_names:
        print("processing {}".format(fn))
        pil_img = PIL.Image.open(fn)
        # convert grayscale
        img = np.array(pil_img.convert('L'))
##        img = cv2.imread(fn, 0)
        h, w = img.shape[:2]

        if h_old:
            assert h == h_old, "h={}, h_old={}".format(h, h_old)
            assert w == w_old
        # corners: (54, 1, 2), coords of chessboard corners in pixel
        # coordinates of `img`
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            # corners_fine: scale up here and refine on scaled coords,
            # should gain some small percent in precision, but not much
            corners_fine = corners.copy() * img_points_scale
            cv2.cornerSubPix(img, corners_fine, (5, 5), (-1, -1), term)
##            cv2.drawChessboardCorners(img, pattern_size, corners, found)
##            cv2.imshow('img',img)
##            cv2.waitKey(5000)
            # corners_fine.reshape(-1,2).shape == (54,2)
            # list of (54,2) arrays, coords of chessboard corners in original
            # image pixel coords scale (img_points_scale applied)
            img_points_lst.append(corners_fine.reshape(-1, 2))
            h_old, w_old = h, w
            # XXX useless, list of the same array over and over!! optimize!!!
            pattern_points_lst.append(pattern_points)
        else:
            print('chessboard not found')
            continue
        
   
    # XXX only if we use drawChessboardCorners() !!!
    cv2.destroyAllWindows()    

    if len(pattern_points_lst) > 0:
        print("calibrateCamera")
        # int stuff only when scaling, could also simply use the orig w and h of
        # the orig images :)
        size = (int(round(w*scale_w)), int(round(h*scale_h)))
        # CALIB_RATIONAL_MODEL: more complex model with 8 parameters instead of 5
        # (default), gives much better results
        rms, camera_matrix, coeffs, rvecs, tvecs = cv2.calibrateCamera(pattern_points_lst,
                                                                       img_points_lst, 
                                                                       size,
                                                                       None,
                                                                       None, None,
                                                                       None,
                                                                       cv2.CALIB_RATIONAL_MODEL,
                                                                       term)
        # XXX not sure about the math here, also only operations on numpy arrays
        # below, make simpler
        print("calculate error")
        mean_error = 0
        for i in range(len(pattern_points_lst)):
            imgpoints2, _ = cv2.projectPoints(pattern_points_lst[i], rvecs[i], tvecs[i], camera_matrix, coeffs)
            error = cv2.norm(img_points_lst[i],imgpoints2.reshape(-1,2), cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        mean_error = mean_error / len(pattern_points_lst)
        
        return {'camera_matrix': camera_matrix, 'rms': rms, 'coeffs': coeffs,
                'mean_error': mean_error}
    else:
        return None

def apply(img_names, dr='converted'):
    # EXIF: cv2.imread() / cv2.imwrite() use plain numpy 3d arrays, we need
    # to fiddle around w/ PIL to extract and add back the EXIF data
    # (orientation, date, camera model, etc, with orientation being the
    # most important information)

    # XXX use common data storage w/ calibrate() such as .unfish/ or simply put
    # all infos in a dict and pickle to disk, or use a hdf5 file
    camera_matrix = np.load('camera_matrix.npy')
    coeffs = np.load('coeffs.npy')

    if not os.path.exists(dr):
        os.makedirs(dr)

    cm = {}
    for ifn,fn in enumerate(img_names):
        tgt = pj(dr, os.path.basename(fn))
        if os.path.exists(tgt):
            print("skip: {}".format(fn))
            continue
        pil_img = PIL.Image.open(fn)
        src = np.array(pil_img)
        hh,ww = src.shape[:2]
        size_src = (ww, hh)
        print("{} {}".format(fn, size_src))
        if not cm.get(size_src,None):
            camera_matrix_new, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                                   coeffs, size_src, 0,
                                                                   size_src)
            print("new camera matrix for size: {}".format(size_src))
            cm[size_src] = {'cm': camera_matrix_new,
                            'roi': roi}
        mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, coeffs, 
                                                 None, cm[size_src]['cm'], size_src,
                                                 cv2.CV_32FC1)
        
        im = PIL.Image.fromarray(cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR))
        im.save(tgt, exif=pil_img.info["exif"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
  
    subparsers = parser.add_subparsers(dest='sub')

    p_prepare = subparsers.add_parser('prep', 
                                       help='prepare calibration run, '
                                            'calculate rms error per calib image')
    p_prepare.add_argument('files', nargs='+')
    
    p_calibrate = subparsers.add_parser('calib', 
                                        help='calibrate using chessboard '
                                             'images')
    p_calibrate.add_argument('files', nargs='+')
    p_calibrate.add_argument('-r', '--max-rms', default=None, type=float,
                             help='maximal allowed RMS error for calib image')
     
  
    p_apply = subparsers.add_parser('apply', 
                                    help='apply corrections to images'
                                         'images')
    p_apply.add_argument('files', nargs='+')
    p_apply.add_argument('dir', default='corrected', nargs='?', 
                         help='target dir for corrected images [%(default)s]')
   
    for sub in [p_prepare, p_calibrate]:
        sub.add_argument('-f', '--fraction', default=1.0, type=float, 
                         help='factor by which calib images are smaller '
                              'than to-be-processed real images '
                              '[%(default)s]')
  
    args = parser.parse_args(sys.argv[1:])
    # chessboard pics for calibration
    # ./this.py calibrate chess_pics/small/*
    # ./this.py apply real_pics/orig/* 
    
    if args.sub == 'calib':
        if args.max_rms:
            with open('rms_db.json') as fd:
                rms_db = json.load(fd)
            files = [k for k,v in rms_db.items() if v < args.max_rms] 
        else:
            files = args.files
        ret = calibrate(files, fraction=args.fraction)
        if ret:
            print "mean error: ", ret['mean_error']
            print "RMS:", ret['rms']
            np.save('camera_matrix.npy', ret['camera_matrix'])
            np.save('coeffs.npy', ret['coeffs'])
        else:
            print("unable to process data")
    elif args.sub == 'prep':
        rms_db = {}
        for fn in args.files:
            ret = calibrate([fn], fraction=args.fraction)
            if ret:
                rms_db[fn] = ret['rms']
                print("{fn}: {rms}".format(fn=fn, rms=ret['rms']))
            else:
                print("#{fn}: no data".format(fn=fn))
        with open('rms_db.json', 'w') as fd:
            json.dump(rms_db, fd, indent=2)
    elif args.sub == 'apply':
        apply(args.files, dr=args.dir)

