from setuptools import setup
import sys
import os
import subprocess
from os.path import join, dirname

#using "http://pythonhosted.org/an_example_pypi_project/setuptools.html"'s setup.py as a template

"""
Maybe useful?
    http://stackoverflow.com/questions/20194565/running-custom-setuptools-build-during-install
    http://www.niteoweb.com/blog/setuptools-run-custom-code-during-install
    http://stackoverflow.com/questions/11778182/how-do-i-add-a-custom-build-step-to-my-setuptools-distribute-setup-py
    https://lucumr.pocoo.org/2012/6/22/hate-hate-hate-everywhere/
    http://parijatmishra.wordpress.com/2008/10/08/python-packaging-setuptools-and-eggs/
    http://svn.python.org/projects/sandbox/trunk/setuptools/setuptools.txt
    http://python-notes.curiousefficiency.org/en/latest/pep_ideas/core_packaging_api.html
"""

# TODO: Account for flags before calling cmake build scripts
# TODO: Account for mingw_build.bat
# TODO: Learn how to work with setup.py build and setup.py install


def read(fname):
    return open(join(dirname(__file__), fname)).read()


if __name__ == '__main__':

    if 'build' in sys.argv:
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
