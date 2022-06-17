#!/usr/bin/env python
r"""
pip install cibuildwheel

Requires OpenCV to build

CIBW_BUILD_VERBOSITY=2 \
CIBW_TEST_REQUIRES="-r requirements/tests.txt" \
CIBW_TEST_COMMAND='python {project}/run_tests.py' \
CIBW_SKIP='pp*' \
    cibuildwheel --config-file pyproject.toml --platform linux --arch x86_64

CIBW_SKIP='pp*' cibuildwheel --config-file pyproject.toml --platform linux --arch x86_64
"""
from __future__ import absolute_import, division, print_function
from os.path import exists

URL_LIST = [
    'http://cmp.felk.cvut.cz/~perdom1/hesaff/',
    'https://github.com/Erotemic/hesaff',
]

ORIGINAL_AUTHORS = 'Krystian Mikolajczyk, Michal Perdoch'
EXTENDED_AUTHORS = 'Jon Crall, Avi Weinstock'


def parse_version(fpath):
    """
    Statically parse the version number from a python file
    """
    import ast
    if not exists(fpath):
        raise ValueError('fpath={!r} does not exist'.format(fpath))
    with open(fpath, 'r') as file_:
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


def parse_description():
    """
    Parse the description in the README file

    CommandLine:
        pandoc --from=markdown --to=rst --output=README.rst README.md
        python -c "import setup; print(setup.parse_description())"
    """
    from os.path import dirname, join, exists
    readme_fpath = join(dirname(__file__), 'README.rst')
    # This breaks on pip install, so check that it exists.
    if exists(readme_fpath):
        with open(readme_fpath, 'r') as f:
            text = f.read()
        return text
    return ''


def parse_requirements(fname='requirements.txt', with_version=False):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if true include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
        python -c "import setup; print(chr(10).join(setup.parse_requirements(with_version=True)))"
    """
    from os.path import exists
    import re
    require_fpath = fname

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


def native_mb_python_tag(plat_impl=None, version_info=None):
    """
    Example:
        >>> print(native_mb_python_tag())
        >>> print(native_mb_python_tag('PyPy', (2, 7)))
        >>> print(native_mb_python_tag('CPython', (3, 8)))
    """
    if plat_impl is None:
        import platform
        plat_impl = platform.python_implementation()

    if version_info is None:
        import sys
        version_info = sys.version_info

    major, minor = version_info[0:2]
    ver = '{}{}'.format(major, minor)

    if plat_impl == 'CPython':
        # TODO: get if cp27m or cp27mu
        impl = 'cp'
        if ver == '27':
            IS_27_BUILT_WITH_UNICODE = True  # how to determine this?
            if IS_27_BUILT_WITH_UNICODE:
                abi = 'mu'
            else:
                abi = 'm'
        else:
            if ver == '38':
                # no abi in 38?
                abi = ''
            else:
                abi = 'm'
        mb_tag = '{impl}{ver}-{impl}{ver}{abi}'.format(**locals())
    elif plat_impl == 'PyPy':
        abi = ''
        impl = 'pypy'
        ver = '{}{}'.format(major, minor)
        mb_tag = '{impl}-{ver}'.format(**locals())
    else:
        raise NotImplementedError(plat_impl)
    return mb_tag


NAME = 'pyhesaff'

VERSION = version = parse_version('pyhesaff/__init__.py')  # needs to be a global var for git tags


INSTALL_REQUIRES = [
    'numpy >= 1.9.0',
    'ubelt',
]

if __name__ == '__main__':
    from skbuild import setup
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

    # import ubelt as ub
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
            (['Release\\*.dll'] if os.name == 'nt' else []) +
            ["LICENSE.txt", "LICENSE-3RD-PARTY.txt", "LICENSE.SIFT"]
    )
    print('pyhesaff_package_data = {!r}'.format(pyhesaff_package_data))

    kwargs = dict(
        name=NAME,
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
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
        },
        maintainer="Jon Crall",
        ext_modules=EmptyListWithLength(),  # hack for including ctypes bins
        install_requires=INSTALL_REQUIRES,
        include_package_data=True,
        # packages=find_packages(include='pyhesaff.*'),
        packages=['pyhesaff'],
        python_requires='>=3.6',
        package_data={
            'pyhesaff': pyhesaff_package_data,
        },
        # package_data={'build': util_cplat.get_dynamic_lib_globstrs()},
        # build_command=lambda: ut.std_build_command(dirname(__file__)),
        classifiers=[
            # List of classifiers available at:
            # https://pypi.python.org/pypi?%3Aaction=list_classifiers
            'Development Status :: 4 - Beta',
            # This should be interpreted as Apache License v2.0
            'License :: OSI Approved :: Apache Software License',
            # Supported Python versions
            # 'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
        ],
    )
    setup(**kwargs)

    # ub.cmd('ls -al', verbose=3)
    # ub.cmd('ls dist', verbose=3)

    print("[setup.py] FINISHING UP")

    if False:
        """
        python -c "import distutils; print(distutils.sys.platform)"
        python -c "import ctypes; print(ctypes.cdll['pyhesaff/libhesaff.win-amd64-3.6.dll'])"
        python -c "import sys, math; print(math.log2(sys.maxsize))"
        """
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
