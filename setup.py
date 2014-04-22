from setuptools import setup
import os
import subprocess

#using "http://pythonhosted.org/an_example_pypi_project/setuptools.html"'s setup.py as a template


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

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
