#!/usr/bin/env python
"""
NOTE: We are using the docker image built using the vtool script now, although
this doesn't give us as wide platform coverage.

Setup the base image containing the opencv deps that pyhesaff needs to build

References:
    https://github.com/skvark/opencv-python
"""
import ubelt as ub
import sys
import os
from os.path import join, exists, realpath


def build_opencv_cmake_args(config):
    """
    Create cmake configuration args for opencv depending on the target platform

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
        unknown_ops = set(config) - set(default_config)
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
            # import subprocess
            # subprocess.check_call(["patch", "-p0", "<", "patches/patchOpenEXR"])

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
        cmake_args.append("-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.8")

    return cmake_args


def build(DPATH, MAKE_CPUS, UNICODE_WIDTH, ARCH, PY_VER, EXEC=True):

    OPENCV_VERSION = '4.1.0'

    dpath = realpath(ub.expandpath(DPATH))
    dpath = ub.ensuredir(dpath)
    os.chdir(dpath)

    BASE = 'manylinux2014_{}'.format(ARCH)
    BASE_REPO = 'quay.io/skvark'

    PY_TAG = 'cp{ver}-cp{ver}m'.format(ver=PY_VER.replace('.', ''))

    # do we need the unicode width in this tag?
    DOCKER_TAG = '{}-opencv{}-py{}'.format(BASE, OPENCV_VERSION, PY_VER)

    if False and EXEC:
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
        'is_64bit': ARCH in {'x86_64'},
        'build_contrib': False,
        'build_headless': True,
        'python_args': {
            'py_ver': PY_VER,
            'py_executable': PY_VER,
        },
        'linux_jpeg_args': {
            'jpeg_include_dir': '${JPEG_INCLUDE_DIR}',
            'jpeg_library': '${JPEG_LIBRARY}',
        }
    }
    CMAKE_ARGS = ' \\\n                '.join(build_opencv_cmake_args(config))

    dockerfile_fpath = join(dpath, 'Dockerfile_' + DOCKER_TAG)
    # This docker code is very specific for building linux binaries.
    # We will need to do a bit of refactoring to handle OSX and windows.
    # But the goal is to get at least one OS working end-to-end.
    docker_code = ub.codeblock(
        r'''
        FROM {BASE_REPO}/{BASE}

        # SETUP ENV
        ARG MB_PYTHON_VERSION={PY_VER}
        ENV _MY_DOCKER_TAG={DOCKER_TAG}
        ENV PYTHON_VERSION={PY_VER}
        ENV PYTHON_ROOT=/opt/python/{PY_TAG}/
        ENV PYTHONPATH=/opt/python/{PY_TAG}/lib/python{PY_VER}/site-packages/
        ENV PATH=/opt/python/{PY_TAG}/bin:$PATH
        ENV PYTHON_EXE=/opt/python/{PY_TAG}/bin/python
        ENV HOME=/root
        ENV ARCH={ARCH}
        ENV UNICODE_WIDTH={UNICODE_WIDTH}

        # Update python environment
        RUN $PYTHON_EXE -m pip install -q --no-cache-dir --upgrade pip  && \
            $PYTHON_EXE -m pip install -q --no-cache-dir ubelt xdoctest cmake ninja scikit-build wheel numpy

        # Obtain opencv source and run platform-specific cmake command
        # TODO: replace this with curl

        RUN $PYTHON_EXE -c "import ubelt; print(ubelt.grabdata('https://github.com/opencv/opencv/archive/{OPENCV_VERSION}.zip', fpath='opencv.zip', hash_prefix='1a00f2cdf2b1bd62e5a700a6f15026b2f2de9b1', hasher='sha512', verbose=0))" && \
            unzip opencv.zip && \
            rm opencv.zip && \
            mkdir -p /root/code && \
            mv opencv-{OPENCV_VERSION} /root/code/opencv && \
            mkdir -p /root/code/opencv/build && \
            cd /root/code/opencv/build && \
            cmake {CMAKE_ARGS} /root/code/opencv && \
            cd /root/code/opencv/build && \
            make -j{MAKE_CPUS} && \
            make install && \
            rm -rf /root/code/opencv
        '''.format(**locals()))

    if EXEC:
        try:
            print(ub.color_text('\n--- DOCKER CODE ---', 'white'))
            print(ub.highlight_code(docker_code, 'docker'))
            print(ub.color_text('--- END DOCKER CODE ---\n', 'white'))
        except Exception:
            pass

    print('...write to dockerfile_fpath = {!r}'.format(dockerfile_fpath))
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

    # write out what the tag is, there might be a better way to pass this
    # information. fixme?
    with open(join(dpath, 'opencv-docker-tag.txt'), 'w') as file:
        file.write(DOCKER_TAG)

    if not EXEC:
        print('-- NO EXEC --')
    else:
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
            print(ub.color_text('\n--- SUCCESS ---', 'green'))

    print('DOCKER_TAG = {!r}'.format(DOCKER_TAG))
    print('dockerfile_fpath = {!r}'.format(dockerfile_fpath))

    # push_cmd = 'docker push quay.io/erotemic/manylinux-opencv:manylinux1_x86_64-opencv4.1.0-py3.6'
    # print('push_cmd = {!r}'.format(push_cmd))
    # print(push_cmd)

    if EXEC:
        QUAY_REPO = 'quay.io/erotemic/manylinux-opencv'
        DOCKER_URI = '{QUAY_REPO}:{DOCKER_TAG}'.format(**locals())
        cmd1 = 'docker tag {DOCKER_TAG} {DOCKER_URI}'.format(**locals())
        cmd2 = 'docker push {DOCKER_URI}'.format(**locals())
        print('-- <push cmds> ---')
        print(cmd1)
        print(cmd2)
        print('-- </push cmds> ---')
        # print('cmd1 = {!r}'.format(cmd1))
        # print('cmd2 = {!r}'.format(cmd2))
        # ub.cmd(cmd1, verbose=3)
        # ub.cmd(cmd2, verbose=3)
        # ub.cmd('docker login quay.io')
        # ub.cmd(push_cmd)

    """
    manylinux-opencv
    """
    return dockerfile_fpath


def main():
    """
    Usage:
        cd ~/code/hesaff/dev
        python ~/code/hesaff/dev/build_opencv_docker.py --allversions --no-exec
        python ~/code/hesaff/dev/build_opencv_docker.py --allversions
    """
    import multiprocessing

    def argval(clikey, envkey=None, default=ub.NoParam):
        if envkey is not None:
            envval = os.environ.get(envkey)
            if envval:
                default = envval
        return ub.argval(clikey, default=default)

    DEFAULT_PY_VER = '{}.{}'.format(sys.version_info.major, sys.version_info.minor)
    DPATH = argval('--dpath', None, default=os.getcwd())
    MAKE_CPUS = argval('--make_cpus', 'MAKE_CPUS', multiprocessing.cpu_count() + 1)
    UNICODE_WIDTH = argval('--unicode_width', 'UNICODE_WIDTH', '32')
    EXEC = not ub.argflag('--no-exec')

    if ub.argflag('--allversions'):
        fpaths = []
        arch = ['i686', 'x86_64', 'aarch64']
        # pyver = ['2.7', '3.4', '3.5', '3.6', '3.7', '3.8']
        pyver = ['3.6', '3.7', '3.8', '3.9']
        for ARCH in arch:
            for PY_VER in pyver:
                fpaths += [build(DPATH, MAKE_CPUS, UNICODE_WIDTH, ARCH, PY_VER, EXEC=EXEC)]

        print("WROTE TO: ")
        print('\n'.join(fpaths))
    else:
        PY_VER = argval('--pyver', 'MB_PYTHON_VERSION', default=DEFAULT_PY_VER)
        ARCH = argval('--arch', 'ARCH', default='x86_64')
        build(DPATH, MAKE_CPUS, UNICODE_WIDTH, ARCH, PY_VER, EXEC=EXEC)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/pyhesaff/dev/build_opencv_docker.py
        python ~/code/pyhesaff/dev/build_opencv_docker.py --no-exec

        docker login quay.io

        docker push quay.io/erotemic/manylinux-opencv:manylinux1_x86_64-opencv4.1.0-py3.6
    """
    main()
