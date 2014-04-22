from setuptools import setup
import os
import subprocess
from os.path import join, dirname

#using "http://pythonhosted.org/an_example_pypi_project/setuptools.html"'s setup.py as a template


# TODO: Account for flags before calling cmake build scripts
# TODO: Account for mingw_build.bat


def read(fname):
    return open(join(dirname(__file__), fname)).read()

subprocess.call(['mkdir', 'build'])

os.chdir('build')

if subprocess.call(['cmake', '-G', 'Unix Makefiles', '..']):
    subprocess.call(['make'])

os.chdir('..')

setup(
    name='pyhesaff',
    version='',
    author='',
    author_email='',
    description=('Routines for computation of hessian affine keypoints in images.'),
    license='',
    keywords='',
    url='https://github.com/Erotemic/hesaff',
    packages=['build', 'pyhesaff', 'tests'],
    package_data={'build': ['*.so']},
    long_description=read('README'),
    classifiers=[
        '',
        '',
        '',
    ],
)
