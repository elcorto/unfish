unfish -- correct fisheye distortions in images using OpenCV

about
-----
This is basically a packaged up, command lined and polished version of the
OpenCV tutorial_ (see also hack_) which shows how to correct lens distortions
in images using OpenCV, based on chessboard calibration images taken with the
same camera. 

In my case, my mobile phone camera introduces a radial distortion (fisheye
effect), hence the name.

There is a script ``bin/unfish`` which does all this and a little more::

	usage:
		unfish prep [-f <fraction> ] <files>...
		unfish calib [-r <max-rms> -f <fraction>] <files>...
		unfish apply [ -d <dir>] <files>...

	commands:
		prep   optional preparation run, create rms_db.json
		calib  calibration run, calculate and write camera matrix and camera model
			   coeffs using chessboard calibration images
		apply  apply correction model to images


	options:
		-f <fraction>, --fraction <fraction>  fraction by which calibration files
				have been scaled down (see makesmall.sh)
		-r <max-rms>, --max-rms <max-rms>  in calibration, use only files with
				rms reprojection error less than <max-rms>, uses rms_db.json
				written by "prep"
		-d <dir>, --dir <dir>  dir where to write corrected images 
				[default: corrected]

In addition to the tutorial_, we added things like the ability to calculate the
RMS reprojection error per calibration image (``unfish prep``), in order to get
a feeling for the quality of the calibration per image.

workflow
--------

First, you print a chessboard_ and take a bunch of calibration images with the
affected camera. Next, a calibration run will calculate correction parameters
(camera matrix and lens model coefficients). Finally, you apply the correction
to all affected images. 

We found that it is a very good idea to scale down the chessboard
calibration images first. That makes the calibration part *a lot* faster (else
the code which searches for chessboard corners will run forever). 

All scripts are in ``bin/`` and here is what you need to do.

::

    $ makesmall.sh 0.2 chess_pics/orig chess_pics/small
    $ unfish calib -f 0.2 chess_pics/small/*
    $ unfish apply affected_pics/orig/*

.. _tutorial: http://docs.opencv.org/3.3.0/dc/dbb/tutorial_py_calibration.html
.. _hack: https://hackaday.io/project/12384-autofan-automated-control-of-air-flow/log/41862-correcting-for-lens-distortions
.. _chessboard: https://github.com/opencv/opencv/blob/master/samples/data/chessboard.png
