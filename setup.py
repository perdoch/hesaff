#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
# from utool import util_cplat
# from utool import util_setup
# import utool as ut
# from os.path import dirname
from skbuild import setup
# from setuptools import find_packages
# from setuptools import setup


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

ORIGINAL_AUTHORS = 'Krystian Mikolajczyk, Michal Perdoch'
EXTENDED_AUTHORS = 'Jon Crall, Avi Weinstock'


def parse_version(package):
    """
    Statically parse the version number from __init__.py

    CommandLine:
        python -c "import setup; print(setup.parse_version('kwimage'))"
    """
    from os.path import dirname, join, exists
    import ast

    # Check if the package is a single-file or multi-file package
    _candiates = [
        join(dirname(__file__), package + '.py'),
        join(dirname(__file__), package, '__init__.py'),
    ]
    _found = [init_fpath for init_fpath in _candiates if exists(init_fpath)]
    if len(_found) > 0:
        init_fpath = _found[0]
    elif len(_found) > 1:
        raise Exception('parse_version found multiple init files')
    elif len(_found) == 0:
        raise Exception('Cannot find package init file')

    with open(init_fpath) as file_:
        sourcecode = file_.read()
    pt = ast.parse(sourcecode)
    class VersionVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if getattr(target, 'id', None) == '__version__':
                    self.version = node.value.s
    visitor = VersionVisitor()
    visitor.visit(pt)
    return visitor.version


version = parse_version('pyhesaff')  # needs to be a global var for git tags


INSTALL_REQUIRES = [
    'numpy >= 1.9.0',
    'ubelt',
]

if __name__ == '__main__':
    kwargs = dict(
        name='pyhesaff',
        description='Routines for computation of hessian affine keypoints in images.',
        url='https://github.com/Erotemic/hesaff',
        author=ORIGINAL_AUTHORS + ', ' + EXTENDED_AUTHORS,
        author_email='erotemic@gmail.com',
        version=version,
        #packages=util_setup.find_packages(),
        # packages=['pyhesaff', 'build', 'pyhesaff.tests'],
        # version=util_setup.parse_package_for_version('pyhesaff'),
        # licence=util_setup.read_license('LICENSE'),
        # long_description=util_setup.parse_readme('README.md'),
        # long_description=parse_description(),
        # install_requires=parse_requirements('requirements/runtime.txt'),
        # extras_require={
        #     'all': parse_requirements('requirements.txt'),
        #     'tests': parse_requirements('requirements/tests.txt'),
        # },
        install_requires=INSTALL_REQUIRES,
        # packages=find_packages(include='pyhesaff.*'),
        packages=['pyhesaff'],
        # package_data={'build': util_cplat.get_dynamic_lib_globstrs()},
        # build_command=lambda: ut.std_build_command(dirname(__file__)),
        classifiers=[
            # List of classifiers available at:
            # https://pypi.python.org/pypi?%3Aaction=list_classifiers
            'Development Status :: 3 - Alpha',
            # This should be interpreted as Apache License v2.0
            'License :: OSI Approved :: Apache Software License',
            # Supported Python versions
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
    )
    setup(**kwargs)
