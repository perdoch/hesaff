#!/usr/bin/env python
"""
Setup the base image containing the opencv deps that pyhesaff needs to build

References:
    https://github.com/skvark/opencv-python
"""
from __future__ import absolute_import, division, print_function
import ubelt as ub
import sys
import os
from os.path import join, exists, realpath


def build_opencv_cmake_args(config):
    """
    References:
        https://github.com/skvark/opencv-python/blob/master/setup.py

    Ignore:
        config = {
            'build_contrib': True,
            'python_args': {
                'py_executable': '${PYTHON_EXE}',
                'py_ver': '3.6',
            },
            'linux_jpeg_args': {
                'jpeg_include_dir': '${JPEG_INCLUDE_DIR}',
                'jpeg_library': '${JPEG_LIBRARY}',
            }
        }
        build_opencv_cmake_args(config)
    """
    DEFAULT_CONFIG = True
    if DEFAULT_CONFIG:
        import sys
        default_config = {
            'is_64bit': sys.maxsize > 2 ** 32,
            'sys_plat': sys.platform,
            'build_contrib': False,
            'build_headless': True,
            'python_args': {
                'py_ver': '{}.{}'.format(sys.version_info[0], sys.version_info[1]),
                'py_executable': sys.executable,
            },
            'linux_jpeg_args': None
        }
        if all(v in os.environ for v in ('JPEG_INCLUDE_DIR', 'JPEG_LIBRARY')):
            default_config['linux_jpeg_args'] = {
                'jpeg_include_dir': os.environ['JPEG_INCLUDE_DIR'],
                'jpeg_library': os.environ['JPEG_LIBRARY'],
            }
        unknown_ops = ub.dict_diff(config, default_config)
        assert not unknown_ops
        for key, value in default_config.items():
            if key not in config:
                config[key] = value

    WIN32 = config['sys_plat'] == 'win32'
    DARWIN = config['sys_plat'] == 'darwin'
    LINUX = config['sys_plat'].startswith('linux')

    if WIN32:
        generator = "Visual Studio 14" + (" Win64" if config['is_64bit'] else '')
    else:
        generator = 'Unix Makefiles'

    cmake_args = [
        '-G', '"{}"'.format(generator),
        # See opencv/CMakeLists.txt for options and defaults
        "-DBUILD_opencv_apps=OFF",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_PERF_TESTS=OFF",
        "-DBUILD_DOCS=OFF"
    ]
    if config['python_args'] is not None:
        py_config = config['python_args']
        PY_MAJOR = py_config['py_ver'][0]
        cmake_args += [
            # skbuild inserts PYTHON_* vars. That doesn't satisfy opencv build scripts in case of Py3
            "-DPYTHON{}_EXECUTABLE={}".format(PY_MAJOR, py_config['py_executable']),
            "-DBUILD_opencv_python{}=ON".format(PY_MAJOR),

            # When off, adds __init__.py and a few more helper .py's. We use
            # our own helper files with a different structure.
            "-DOPENCV_SKIP_PYTHON_LOADER=ON",
            # Relative dir to install the built module to in the build tree.
            # The default is generated from sysconfig, we'd rather have a constant for simplicity
            "-DOPENCV_PYTHON{}_INSTALL_PATH=python".format(PY_MAJOR),
            # Otherwise, opencv scripts would want to install `.pyd' right into site-packages,
            # and skbuild bails out on seeing that
            "-DINSTALL_CREATE_DISTRIB=ON",
        ]

    if config['build_contrib']:
        # TODO: need to know abspath
        root = '.'
        cmake_args += [
            "-DOPENCV_EXTRA_MODULES_PATH=" + join(root, "opencv_contrib/modules")
        ]

    if config['build_headless']:
        # it seems that cocoa cannot be disabled so on macOS the package is not truly headless
        cmake_args.append("-DWITH_WIN32UI=OFF")
        cmake_args.append("-DWITH_QT=OFF")
    else:
        if DARWIN or LINUX:
            cmake_args.append("-DWITH_QT=4")

    if LINUX:
        cmake_args.append("-DWITH_V4L=ON")
        cmake_args.append("-DENABLE_PRECOMPILED_HEADERS=OFF")

        # tests fail with IPP compiled with
        # devtoolset-2 GCC 4.8.2 or vanilla GCC 4.9.4
        # see https://github.com/skvark/opencv-python/issues/138
        cmake_args.append("-DWITH_IPP=OFF")
        if not config['is_64bit']:
            cmake_args.append("-DCMAKE_CXX_FLAGS=-U__STRICT_ANSI__")

        if config['linux_jpeg_args'] is not None:
            jpeg_config = config['linux_jpeg_args']
            cmake_args += [
                "-DBUILD_JPEG=OFF",
                "-DJPEG_INCLUDE_DIR=" + jpeg_config['jpeg_include_dir'],
                "-DJPEG_LIBRARY=" + jpeg_config['jpeg_library'],
            ]

    # Fixes for macOS builds
    if DARWIN:
        # Some OSX LAPACK fns are incompatible, see
        # https://github.com/skvark/opencv-python/issues/21
        cmake_args.append("-DWITH_LAPACK=OFF")
        cmake_args.append("-DCMAKE_CXX_FLAGS=-stdlib=libc++")
        cmake_args.append("-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.7")

    return ' '.join(cmake_args)


def main():

    def argval(clikey, envkey=None, default=ub.NoParam):
        if envkey is not None:
            envval = os.environ.get(envkey)
            if envval:
                default = envval
        return ub.argval(clikey, default=default)

    DEFAULT_PY_VER = '{}.{}'.format(sys.version_info.major, sys.version_info.minor)
    PY_VER = argval('--pyver', 'MB_PYTHON_VERSION', default=DEFAULT_PY_VER)

    dpath = argval('--dpath', None, default=os.getcwd())
    PLAT = argval('--plat', 'PLAT', default='x86_64')

    UNICODE_WIDTH = argval('--unicode_width', 'UNICODE_WIDTH', '32')

    import multiprocessing
    MAKE_CPUS = argval('--make_cpus', 'MAKE_CPUS', multiprocessing.cpu_count() + 1)

    OPENCV_VERSION = '4.1.0'

    dpath = realpath(ub.expandpath(dpath))
    dpath = ub.ensuredir(dpath)
    os.chdir(dpath)

    BASE = 'manylinux1_{}'.format(PLAT)
    BASE_REPO = 'quay.io/skvark'

    PY_TAG = 'cp{ver}-cp{ver}m'.format(ver=PY_VER.replace('.', ''))

    # do we need the unicode width in this tag?
    DOCKER_TAG = '{}-opencv{}-py{}'.format(BASE, OPENCV_VERSION, PY_VER)

    if not exists(join(dpath, 'opencv-' + OPENCV_VERSION)):
        # FIXME: make robust in the case this fails
        print('downloading opencv')
        fpath = ub.grabdata(
            'https://github.com/opencv/opencv/archive/{}.zip'.format(OPENCV_VERSION),
            dpath=dpath, hash_prefix='1a00f2cdf2b1bd62e5a700a6f15026b2f2de9b1',
            hasher='sha512', verbose=0
        )
        ub.cmd('ln -s {} .'.format(fpath), cwd=dpath, verbose=0)
        ub.cmd('unzip {}'.format(fpath), cwd=dpath, verbose=0)

    config = {
        'is_64bit': PLAT in {'x86_64'},
        'build_contrib': False,
        'build_headless': True,
        'python_args': None,
        # {
        #     'py_ver': PY_VER,
        #     'py_executable': PY_VER,
        # },
        'linux_jpeg_args': {
            'jpeg_include_dir': '${JPEG_INCLUDE_DIR}',
            'jpeg_library': '${JPEG_LIBRARY}',
        }
    }
    CMAKE_ARGS = build_opencv_cmake_args(config)

    dockerfile_fpath = join(dpath, 'Dockerfile_' + DOCKER_TAG)
    # This docker code is very specific for building linux binaries.
    # We will need to do a bit of refactoring to handle OSX and windows.
    # But the goal is to get at least one OS working end-to-end.
    docker_code = ub.codeblock(
        '''
        FROM {BASE_REPO}/{BASE}

        # SETUP ENV
        ARG MB_PYTHON_VERSION={PY_VER}
        ENV PYTHON_VERSION={PY_VER}
        ENV PYTHON_ROOT=/opt/python/{PY_TAG}/
        ENV PYTHONPATH=/opt/python/{PY_TAG}/lib/python{PY_VER}/site-packages/
        ENV PATH=/opt/python/{PY_TAG}/bin:$PATH
        ENV PYTHON_EXE=/opt/python/{PY_TAG}/bin/python
        ENV HOME=/root
        ENV PLAT={PLAT}
        ENV UNICODE_WIDTH={UNICODE_WIDTH}

        # Update python environment
        RUN echo "$PYTHON_EXE"
        RUN $PYTHON_EXE -m pip install --upgrade pip && \
            $PYTHON_EXE -m pip install cmake ninja scikit-build wheel numpy

        # This is very different for different operating systems
        # https://github.com/skvark/opencv-python/blob/master/setup.py
        COPY opencv-{OPENCV_VERSION} /root/code/opencv
        # RUN mkdir -p /root/code/opencv/build && \
        #     cd /root/code/opencv/build && \
        #     cmake -G "Unix Makefiles" \
        #            -DINSTALL_CREATE_DISTRIB=ON \
        #            -DOPENCV_SKIP_PYTHON_LOADER=ON \
        #            -DBUILD_opencv_apps=OFF \
        #            -DBUILD_SHARED_LIBS=OFF \
        #            -DBUILD_TESTS=OFF \
        #            -DBUILD_PERF_TESTS=OFF \
        #            -DBUILD_DOCS=OFF \
        #            -DWITH_QT=OFF \
        #            -DWITH_IPP=OFF \
        #            -DWITH_V4L=ON \
        #            -DBUILD_JPEG=OFF \
        #            -DENABLE_PRECOMPILED_HEADERS=OFF \
        #         /root/code/opencv

        RUN mkdir -p /root/code/opencv/build && \
            cd /root/code/opencv/build && \
            cmake {CMAKE_ARGS} /root/code/opencv

        # Note: there is no need to compile the above with python
        # -DPYTHON3_EXECUTABLE=$PYTHON_EXE \
        # -DBUILD_opencv_python3=ON \
        # -DOPENCV_PYTHON3_INSTALL_PATH=python \

        RUN cd /root/code/opencv/build && make -j{MAKE_CPUS} && make install
        '''.format(**locals()))

    if 1:
        try:
            print(ub.color_text('\n--- DOCKER CODE ---', 'white'))
            print(ub.highlight_code(docker_code, 'docker'))
            print(ub.color_text('--- END DOCKER CODE ---\n', 'white'))
        except Exception:
            pass
    with open(dockerfile_fpath, 'w') as file:
        file.write(docker_code)

    docker_build_cli = ' '.join([
        'docker', 'build',
        # '--build-arg PY_VER={}'.format(PY_VER),
        '--tag {}'.format(DOCKER_TAG),
        '-f {}'.format(dockerfile_fpath),
        '.'
    ])
    print('docker_build_cli = {!r}'.format(docker_build_cli))

    print('EXEC DOCKER')
    info = ub.cmd(docker_build_cli, verbose=3, shell=True)

    if info['ret'] != 0:
        print(ub.color_text('\n--- FAILURE ---', 'red'))
        print('Failed command:')
        print(info['command'])
        print(info['err'])
        print('NOTE: sometimes reruning the command manually works')
        raise Exception('Building docker failed with exit code {}'.format(info['ret']))
    else:
        # write out what the tag is
        with open(join(dpath, 'opencv-docker-tag.txt'), 'w') as file:
            file.write(DOCKER_TAG)
        print(ub.color_text('\n--- SUCCESS ---', 'green'))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/hesaff/docker_build.py
    """
    main()
