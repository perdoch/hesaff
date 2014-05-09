from setuptools import setup
import sys
import os
import subprocess
from os.path import join, dirname, exists, split

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

__SETUP_DIR__ = dirname(__file__)
__BUILD_DIR__ = join(__SETUP_DIR__, 'build')
__CWD__ = os.getcwd()


def assert_in_hesaff_repo():
    print('__CWD__       = %r' % (__CWD__))
    print('__SETUP_DIR__ = %r' % (__SETUP_DIR__))
    print('__BUILD_DIR__ = %r' % (__BUILD_DIR__))

    repo_dname = split(__SETUP_DIR__)[1]

    print('repo_dname = %r' % repo_dname)

    try:
        assert repo_dname == 'hesaff'
        assert __CWD__ == __SETUP_DIR__
        assert exists(__SETUP_DIR__)
        assert exists(join(__SETUP_DIR__, 'setup.py'))
        assert exists(join(__SETUP_DIR__, 'pyhesaff'))
    except AssertionError as ex:
        print(ex)
        print('ERROR!: NOT IN HESAFF REPO')
        raise


def read_from(fname):
    with open(join(__SETUP_DIR__, fname)) as file_:
        return file_.read()


def build():
    os.chdir(__SETUP_DIR__)
    if sys.platform.startswith('win32'):
        subprocess.call(['mingw_hesaff_build.bat'])
    else:
        subprocess.call(['unix_hesaff_build.sh'])

if __name__ == '__main__':

    assert_in_hesaff_repo()

    if 'build' in sys.argv or not exists(__BUILD_DIR__):
        build()

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
        long_description=read_from('README'),
        classifiers=[
            '',
            '',
            '',
        ],
    )
