# publish on pypi
# ---------------
#   $ python3 setup.py sdist
#   $ twine upload dist/unfish-x.y.z.tar.gz

import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
bindir = 'bin'
with open(os.path.join(here, 'README.rst')) as fd:
    long_description = fd.read()


setup(
    name='unfish',
    version='0.0.0',
    description='correct fisheye distortions in images using OpenCV',
    long_description=long_description,
    url='https://github.com/elcorto/unfish',
    author='Steve Schmerler',
    author_email='git@elcorto.com',
    license='BSD 3-Clause',
    keywords='camera fisheye opencv',
    packages=['unfish'],
    install_requires=open('requirements.txt').read().splitlines(),
    scripts=['{}/{}'.format(bindir, script) for script in os.listdir(bindir)]
)
