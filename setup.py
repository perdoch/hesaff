#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from skbuild import setup

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
    import sysconfig
    import os
    import sys
    class EmptyListWithLength(list):
        def __len__(self):
            return 1

    try:
        soconfig = sysconfig.get_config_var('EXT_SUFFIX')
    except Exception:
        soconfig = sysconfig.get_config_var('SO')

    def get_lib_ext():
        if sys.platform.startswith('win32'):
            ext = '.dll'
        elif sys.platform.startswith('darwin'):
            ext = '.dylib'
        elif sys.platform.startswith('linux'):
            ext = '.so'
        else:
            raise Exception('Unknown operating system: %s' % sys.platform)
        return ext

    libext = get_lib_ext()

    if True:
        # _os = 'linux'
        # assert _os == 'linux'
        # _arch = 'x86_64'
        # _pyver = '3.6'
        _pyver = '{}.{}'.format(sys.version_info.major, sys.version_info.minor)
        hack_libconfig = '-{}{}'.format(_pyver, libext)
        # hack_libconfig = '.{}-{}-{}.so'.format(_os, _arch, _pyver)
        # hack_libconfig = '.so'
        # print('hack_libconfig = {!r}'.format(hack_libconfig))

    import ubelt as ub
    # ub.cmd('pwd', verbose=3)
    # ub.cmd('ls -al', verbose=3)
    # ub.cmd('ls dist', verbose=3)
    # ub.cmd('ls -al pyhesaff', verbose=3)

    pyhesaff_package_data = (
            ['*%s' % soconfig] +
            ['*%s' % hack_libconfig] +
            ['*%s' % libext] +
            # ['*.so'] +
            (['*.dll'] if os.name == 'nt' else []) +
            ["LICENSE.txt", "LICENSE-3RD-PARTY.txt", "LICENSE.SIFT"]
    )
    print('pyhesaff_package_data = {!r}'.format(pyhesaff_package_data))

    kwargs = dict(
        name='pyhesaff',
        description='Routines for computation of hessian affine keypoints in images.',
        url='https://github.com/Erotemic/hesaff',
        author=ORIGINAL_AUTHORS + ', ' + EXTENDED_AUTHORS,
        author_email='erotemic@gmail.com',
        version=version,
        # license=['Apache 2', util_setup.read_license('LICENSE.SIFT')],
        # long_description=util_setup.parse_readme('README.md'),
        # long_description_content_type='text/x-rst',
        # long_description_content_type='text/markdown',
        # install_requires=parse_requirements('requirements/runtime.txt'),
        # extras_require={
        #     'all': parse_requirements('requirements.txt'),
        #     'tests': parse_requirements('requirements/tests.txt'),
        # },
        maintainer="Jon Crall",
        ext_modules=EmptyListWithLength(),  # hack for including ctypes bins
        install_requires=INSTALL_REQUIRES,
        include_package_data=True,
        # packages=find_packages(include='pyhesaff.*'),
        packages=['pyhesaff'],
        package_data={
            'pyhesaff': pyhesaff_package_data,
        },
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

    # ub.cmd('ls -al', verbose=3)
    ub.cmd('ls dist', verbose=3)

    print("[setup.py] FINISHING UP")

    if True:
        import sys
        if '--inplace' in sys.argv:
            def get_plat_specifier():
                """
                Standard platform specifier used by distutils
                """
                import distutils
                try:
                    plat_name = distutils.util.get_platform()
                except AttributeError:
                    plat_name = distutils.sys.platform
                plat_spec = ".%s-%s" % (plat_name, sys.version[0:3])
                if hasattr(sys, 'gettotalrefcount'):
                    plat_spec += '-pydebug'
                return plat_spec
            print("DOING INPLACE HACK")
            # HACK: I THINK A NEW SCIKIT-BUILD WILL FIX THIS
            import os
            from os.path import join
            spec = get_plat_specifier()
            dspec = spec.lstrip('.')

            if sys.platform.startswith('win32'):
                src = join(os.getcwd(), '_skbuild/{}/cmake-build/Release/hesaff{}'.format(dspec, libext))
                dst = join(os.getcwd(), 'pyhesaff/libhesaff{}{}'.format(spec, libext))
            else:
                src = join(os.getcwd(), '_skbuild/{}/cmake-build/libhesaff{}'.format(dspec, libext))
                dst = join(os.getcwd(), 'pyhesaff/libhesaff{}{}'.format(spec, libext))
            import shutil
            print('copy {} -> {}'.format(src, dst))
            shutil.copy(src, dst)

    print("[setup.py] FINISHED")
