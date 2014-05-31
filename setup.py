#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from utool.util_setup import setuptools_setup
from utool import util_cplat


def build_command():
    """ Build command run by utool.util_setup """
    print('Running custom build')
    if util_cplat.WIN32:
        util_cplat.shell('mingw_build.bat')
    else:
        util_cplat.shell('./unix_build.sh')


URL_LIST = [
    'http://cmp.felk.cvut.cz/~perdom1/hesaff/',
    'https://github.com/Erotemic/hesaff',
]


INSTALL_REQUIRES = [
    'numpy >= 1.8.0',
]

if __name__ == '__main__':

    setup_dict = {
        'name':             'pyhesaff',
        'build_command':    build_command,
        'description':      'Routines for computation of hessian affine keypoints in images.',
        'url':              'https://github.com/Erotemic/hesaff',
        'author':           'Krystian Mikolajczyk, Michal Perdoch, Jon Crall, Avi Weinstock',
        'author_email':     'erotemic@gmail.com',
        'packages':         ['build', 'pyhesaff'],
        'install_requires': INSTALL_REQUIRES,
        'package_data':     {'build': util_cplat.get_dynamic_lib_globstrs()},
        'setup_fpath':      __file__,
    }
    setuptools_setup(**setup_dict)
