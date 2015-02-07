#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from utool import util_cplat
from utool import util_setup
import utool as ut
from setuptools import setup


#def build_command():
#    """ Build command run by utool.util_setup
#    TODO: can use ut.std_build_cmd instead
#    """
#    print('Running custom build')
#    if util_cplat.WIN32:
#        util_cplat.shell('mingw_build.bat')
#    else:
#        util_cplat.shell('./unix_build.sh')


URL_LIST = [
    'http://cmp.felk.cvut.cz/~perdom1/hesaff/',
    'https://github.com/Erotemic/hesaff',
]


INSTALL_REQUIRES = [
    'numpy >= 1.8.0',
]

if __name__ == '__main__':
    setup_dict = dict(
        name='pyhesaff',
        #packages=util_setup.find_packages(),
        packages=['pyhesaff', 'build', 'pyhesaff.tests'],
        version=util_setup.parse_package_for_version('pyhesaff'),
        licence=util_setup.read_license('LICENSE'),
        long_description=util_setup.parse_readme('README.md'),
        description='Routines for computation of hessian affine keypoints in images.',
        url='https://github.com/Erotemic/hesaff',
        author='Krystian Mikolajczyk, Michal Perdoch, Jon Crall, Avi Weinstock',
        author_email='erotemic@gmail.com',
        install_requires=INSTALL_REQUIRES,
        package_data={'build': util_cplat.get_dynamic_lib_globstrs()},
        build_command=ut.std_build_command,
        setup_fpath=__file__,
    )
    kwargs = util_setup.setuptools_setup(**setup_dict)
    setup(**kwargs)
