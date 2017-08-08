#!/usr/bin/python3

"""
usage:
    unfish prep [-f <fraction> ] <files>...
    unfish calib [-r <max-rms> -f <fraction>] <files>...
    unfish apply [ -d <dir>] <files>...


options:
    -f <fraction>, --fraction <fraction>  fraction by which calibration files
            have been scaled down (see makesmall.sh)
    -r <max-rms>, --max-rms <max-rms>  in calibration, use only files with
            rms reprojection error less than <max-rms>, uses rms_db.json
            written by "prep"
    -d <dir>, --dir <dir>  dir where to write corrected images 
            [default: corrected]
"""

import numpy as np
import cv2, docopt
import os, sys, json
import PIL.Image
pj = os.path.join

# https://hackaday.io/project/12384-autofan-automated-control-of-air-flow/log/41862-correcting-for-lens-distortions
# http://docs.opencv.org/3.3.0/dc/dbb/tutorial_py_calibration.html
# https://codeyarns.com/2014/01/16/how-to-convert-between-numpy-array-and-pil-image/

# numpy: shape = (hh, ww) = (nrows, ncols)
# opencv: size = (ww, hh)

def calibrate(img_names, fraction=0.2, maxiter=30, tol=0.1, pattern_size=(9,6)):
    """
    Parameters
    ----------
    img_names : sequence
        list of calibration image files
    fraction : float
        factor by which calibration images are scaled down 
        by, e.g. makesmall.sh
    maxiter, tol : termination criteria for findChessboardCorners and
        calibrateCamera, see opencv docs
    pattern_size : tuple
        size of chessboard pattern (number of corners)

    Notes
    -----
    termination criteriam : maxiter=1000 and tol=0.0001 makes results worse
    again (too much counter-bending at the image corners) .. strange
    
    fraction : original images scaled down by factor `fraction` because
    calibration is 
    MUCH faster and the accuracy of findChessboardCorners on the scaled down
    calibration images is by far enough. Difference of 1/fraction to exact
    scale factors based on image size: example w/ actual scaled image sizes
    for fraction=0.2, so 1/fraction = 5.0:
        3264/653.0 = 4.998468606431853
        2448/490.0 = 4.995918367346939
    """ 
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, maxiter, tol)
    scale_h = scale_w = img_points_scale = 1.0/fraction
    
    # XXX quite involved numpyology here, can this be dome in a simpler way??
    # rectangular grid of chessboard corners, viewed along chessboard plane
    # normal vector; z coord is zero; the "real" object = the undistorted chessboard
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32 )
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    
    # XXX get rid of pattern_points_lst, it a list of the same numpy array
    pattern_points_lst = []
    img_points_lst = []
    shape_old = None
    for fn in img_names:
        print("processing {}".format(fn))
        # PIL: convert('L') = convert grayscale, PIL gives us the same numpy
        # array shape regardless of EXIF orientation, while cv2.imread()
        # rotates the images
        img = np.array(PIL.Image.open(fn).convert('L'))
        shape = img.shape[:2]
        hh, ww = shape
        if shape_old:
            assert shape == shape_old, ("{} != old {}".format(shape, shape_old))
        
        # corners: (54, 1, 2) for pattern_points=(9,6), coords of chessboard
        # corners in pixel coordinates of `img`
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            # corners_fine: scale up to original size coordinates and refine
            # corners, should gain some small percent in precision, but not
            # much
            corners_fine = corners.copy() * img_points_scale
            cv2.cornerSubPix(img, corners_fine, (5, 5), (-1, -1), term)
##            cv2.drawChessboardCorners(img, pattern_size, corners, found)
##            cv2.imshow('img',img)
##            cv2.waitKey(5000)
            # corners_fine.reshape(-1,2).shape == (54,2)
            # list of (54,2) arrays, coords of chessboard corners in original
            # image pixel coords scale (img_points_scale applied)
            img_points_lst.append(corners_fine.reshape(-1, 2))
            shape_old = shape
            # XXX useless, list of the same array over and over!! optimize!!!
            pattern_points_lst.append(pattern_points)
        else:
            print('chessboard not found')
            continue

##    # XXX only if we use drawChessboardCorners() !!!
##    # not implemented in pip3-installed version of opencv
##    cv2.destroyAllWindows()

    if len(pattern_points_lst) > 0:
        print("calibrateCamera")
        # int(round(...)) only when scaling with img_points_scale, could also
        # simply use the orig w and h of the orig images :)
        size = (int(round(ww*scale_w)), int(round(hh*scale_h)))
        # CALIB_RATIONAL_MODEL: more complex model with 8 parameters (default 5
        # + params k4,k5 and k6) which gives much better results. Apparently
        # the way to enable multiple flags is to add them. With
        # CALIB_THIN_PRISM_MODEL and CALIB_TILTED_MODEL, more parameters are
        # added to the model (s1,s2,s3,s4) and (tau_x and tau_y), which,
        # however, doesn't make the results any better :)
        flags = cv2.CALIB_RATIONAL_MODEL
        ##flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL
        ##flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL
        rms, camera_matrix, coeffs, rvecs, tvecs = cv2.calibrateCamera(pattern_points_lst,
                                                                       img_points_lst,
                                                                       size,
                                                                       None,
                                                                       None, None,
                                                                       None,
                                                                       flags,
                                                                       term)
        return {'camera_matrix': camera_matrix, 'rms': rms, 'coeffs': coeffs}
    else:
        return None


def apply(img_names, dr='converted'):
    """Apply image corrections (revert fisheye). Use camera matrix and model
    coeffs written by :func:`calibrate`.

    Parameters
    ----------
    img_names : sequence
        list of to-be-corrected image files
    dr : str
        directory to which we write the corrected images
    """
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
    args = docopt.docopt(__doc__, options_first=False)

    if args['calib']:
        if args['--max-rms']:
            with open('rms_db.json') as fd:
                rms_db = json.load(fd)
            files = [k for k,v in rms_db.items() if v < float(args['--max-rms'])]
        else:
            files = args['<files>']
        ret = calibrate(files, fraction=float(args['--fraction']))
        if ret:
            print("rms: {}".format(ret['rms']))
            np.save('camera_matrix.npy', ret['camera_matrix'])
            np.save('coeffs.npy', ret['coeffs'])
        else:
            print("unable to process data")
    elif args['prep']:
        rms_db = {}
        for fn in args['<files>']:
            ret = calibrate([fn], fraction=float(args['--fraction']))
            if ret:
                rms_db[fn] = ret['rms']
                print("{fn}: {rms}".format(fn=fn, rms=ret['rms']))
            else:
                print("#{fn}: no data".format(fn=fn))
        with open('rms_db.json', 'w') as fd:
            json.dump(rms_db, fd, indent=2)
    elif args['apply']:
        apply(args['<files>'], dr=args['--dir'])

