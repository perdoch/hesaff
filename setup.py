#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from utool.util_setup import setuptools_setup
from utool.util_cplat import get_dynamic_lib_globstrs
import sys
import subprocess
import pyhesaff


def build_command():
    """ Build command run by utool.util_setup """
    if sys.platform.startswith('win32'):
        subprocess.call(['mingw_hesaff_build.bat'])
    else:
        subprocess.call(['unix_hesaff_build.sh'])


URL_LIST = [
    'http://cmp.felk.cvut.cz/~perdom1/hesaff/',
    'https://github.com/Erotemic/hesaff',
]

if __name__ == '__main__':
    setuptools_setup(
        setup_fpath=__file__,
        module=pyhesaff,
        build_command=build_command,
        description=('Routines for computation of hessian affine keypoints in images.'),
        url='https://github.com/Erotemic/hesaff',
        author='Krystian Mikolajczyk, Michal Perdoch, Jon Crall, Avi Weinstock',
        author_email='erotemic@gmail.com',
        packages=['build', 'pyhesaff'],
        py_modules=['pyhesaff'],
        package_data={'build': get_dynamic_lib_globstrs()},
    )
