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
        RUN mkdir -p /root/code/opencv/build && \
            cd /root/code/opencv/build && \
            cmake -G "Unix Makefiles" \
                   -DINSTALL_CREATE_DISTRIB=ON \
                   -DOPENCV_SKIP_PYTHON_LOADER=ON \
                   -DBUILD_opencv_apps=OFF \
                   -DBUILD_SHARED_LIBS=OFF \
                   -DBUILD_TESTS=OFF \
                   -DBUILD_PERF_TESTS=OFF \
                   -DBUILD_DOCS=OFF \
                   -DWITH_QT=OFF \
                   -DWITH_IPP=OFF \
                   -DWITH_V4L=ON \
                   -DBUILD_JPEG=OFF \
                   -DENABLE_PRECOMPILED_HEADERS=OFF \
                /root/code/opencv

        # Note: there is no need to compile the above with python
        # -DPYTHON3_EXECUTABLE=$PYTHON_EXE \
        # -DBUILD_opencv_python3=ON \
        # -DOPENCV_PYTHON3_INSTALL_PATH=python \

        RUN cd /root/code/opencv/build && make -j{MAKE_CPUS} && make install
        '''.format(**locals()))

    if 0:
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

    try:
        from xdoctest.utils import strip_ansi
        info = ub.cmd(docker_build_cli, verbose=0, shell=True)
        print(strip_ansi(info['out']))
        print(strip_ansi(info['err']))
    except Exception:
        print('EXEC DOCKER')
        info = ub.cmd(docker_build_cli, verbose=0, shell=True)

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
