import numpy as np
from itertools import product
import cv2, os, PIL.Image

# numpy: shape = (hh, ww) = (nrows, ncols)
# opencv: size = (ww, hh)

def calibrate(img_names, fraction=0.2, maxiter=30, tol=0.1, pattern_size=(9,6),
              disp=False):
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
    disp : bool
        display found chessboard corners during calibration

    Notes
    -----
    termination criteriam : maxiter=1000 and tol=0.0001 makes results worse
    again (too much counter-bending at the image corners) .. strange
    
    fraction : We usually scale the original calibration images down by factor
    `fraction` because calibration is MUCH faster and the accuracy of
    findChessboardCorners on the scaled down calibration images is by far
    enough. Difference of 1/fraction to exact scale factors based on image
    size: example w/ actual scaled image sizes for fraction=0.2, so 1/fraction
    = 5.0:
        3264/653.0 = 4.998468606431853
        2448/490.0 = 4.995918367346939
    """ 
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, maxiter, tol)
    scale_h = scale_w = img_points_scale = 1.0/fraction
    
    # pattern_points.shape = (54,3), pattern_points[:,2] = 0.0: rectangular
    # grid of chessboard corners (54,2), viewed along chessboard plane
    # normal vector; z coord is zero; the "real" object = the undistorted
    # chessboard
    itrs = ([0], range(pattern_size[1]), range(pattern_size[0]))
    pattern_points = np.array(list(product(*itrs)), np.float32)[:,::-1]
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
            if disp:
                cv2.drawChessboardCorners(img, pattern_size, corners, found)
                cv2.imshow('img',img)
                cv2.waitKey(5000)
            # corners_fine.reshape(-1,2).shape == (54,2)
            # list of (54,2) arrays, coords of chessboard corners in original
            # image pixel coords scale (img_points_scale applied)
            img_points_lst.append(corners_fine.reshape(-1, 2))
            shape_old = shape
            # Yes, we append the same array over and over, yes this is
            # ridiculous, but the API of calibrateCamera() wants it that way.
            # We could also count the number of valid calibration images or
            # smth ...
            pattern_points_lst.append(pattern_points)
        else:
            print('chessboard not found')
            continue

    if disp:
        # not implemented in pip3-installed version of opencv
        cv2.destroyAllWindows()

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
        ##flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL + \
        ##    cv2.CALIB_TILTED_MODEL
        rms, camera_matrix, coeffs, rvecs, tvecs = \
                cv2.calibrateCamera(pattern_points_lst,
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
        tgt = os.path.join(dr, os.path.basename(fn))
        if os.path.exists(tgt):
            print("skip: {}".format(fn))
            continue
        pil_img = PIL.Image.open(fn)
        src = np.array(pil_img)
        hh,ww = src.shape[:2]
        size_src = (ww, hh)
        print("{} {}".format(fn, size_src))
        if not cm.get(size_src,None):
            camera_matrix_new, roi = \
                    cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                  coeffs, size_src, 0,
                                                  size_src)
            print("new camera matrix for size: {}".format(size_src))
            cm[size_src] = {'cm': camera_matrix_new,
                            'roi': roi}
        mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, coeffs,
                                                 None, cm[size_src]['cm'], 
                                                 size_src,
                                                 cv2.CV_32FC1)

        im = PIL.Image.fromarray(cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR))
        im.save(tgt, exif=pil_img.info["exif"])
