# publish on pypi
# ---------------
#   $ python3 setup.py sdist
#   $ twine upload dist/unfish-x.y.z.tar.gz 

import os, importlib
from setuptools import setup
from distutils.version import StrictVersion as Version

here = os.path.abspath(os.path.dirname(__file__))
bindir = 'bin'
with open(os.path.join(here, 'README.rst')) as fd:
    long_description = fd.read()

# Hack to make pip respect system packages. So sad! Why in the world can pip
# list and uninstall system packages (Debian: /usr/lib/python3/dist-packages),
# but NOT respect them when installing!!!?? It completely ignores existing
# packages, even with correct versions, and happily re-installs them from
# it's own sources (of course to other locations). Then we end up with two
# installations of the SAME package in the SAME version -- thank you very much.
# I don't get it, seriously.
install_requires = []
req = [('numpy', 'numpy', None, None),
       ('docopt', 'docopt', None, None),
       ('opencv-python', 'cv2', '>=', '3.0')]

for pipname,pkgname,op,ver in req:
    if op and ver:
        this_req = pipname + op + ver
    else:
        this_req = pipname
    try:
        pkg = importlib.import_module(pkgname)
        if op and ver:
            cmd = "Version(pkg.__version__) {op} Version('{ver}')".format(op=op,
                                                                          ver=ver)
            if not eval(cmd):
                install_requires.append(this_req)
    except ImportError:
        install_requires.append(this_req)

setup(
    name='unfish',
    version='0.0.0',
    description='correct fisheye distortions in images using OpenCV',
    long_description=long_description,
    url='https://github.com/elcorto/unfish',
    author='Steve Schmerler',
    author_email='git@elcorto.com',
    license='GPLv3',
    keywords='camera fisheye opencv',
    packages=['unfish'],
    install_requires=install_requires,
    scripts=['{}/{}'.format(bindir, script) for script in os.listdir(bindir)]
)
