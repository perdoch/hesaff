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
from os.path import join, exists


def main():
    dpath = ub.argval('--dpath', default=os.getcwd())
    os.chdir(dpath)

    BASE = 'manylinux1_x86_64'
    BASE_REPO = 'quay.io/skvark'
    OPENCV_VERSION = '4.1.0'
    PY_VER = '{}.{}'.format(sys.version_info.major, sys.version_info.minor)
    tag = '{}-opencv{}-py{}'.format(BASE, OPENCV_VERSION, PY_VER)

    if not exists(join(dpath, 'opencv-' + OPENCV_VERSION)):
        # FIXME: make robust in the case this fails
        fpath = ub.grabdata('https://github.com/opencv/opencv/archive/{}.zip'.format(OPENCV_VERSION), dpath=dpath, verbose=1)
        ub.cmd('ln -s {} .'.format(fpath), cwd=dpath, verbose=3)
        ub.cmd('unzip {}'.format(fpath), cwd=dpath, verbose=3)

    dockerfile_fpath = join(dpath, 'Dockerfile_' + tag)
    # This docker code is very specific for building linux binaries.
    # We will need to do a bit of refactoring to handle OSX and windows.
    # But the goal is to get at least one OS working end-to-end.
    docker_code = ub.codeblock(
        '''
        FROM {BASE_REPO}/{BASE}

        # SETUP ENV
        ARG MB_PYTHON_VERSION=3.6
        ARG ENABLE_CONTRIB=0
        ARG ENABLE_HEADLESS=1
        ENV PYTHON_VERSION=3.6
        ENV PYTHON_ROOT=/opt/python/cp36-cp36m/
        ENV PYTHONPATH=/opt/python/cp36-cp36m/lib/python3.6/site-packages/
        ENV PATH=/opt/python/cp36-cp36m/bin:$PATH
        ENV PYTHON_EXE=/opt/python/cp36-cp36m/bin/python
        ENV HOME=/root
        ENV PLAT=x86_64
        ENV UNICODE_WIDTH=32

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

        RUN cd /root/code/opencv/build && make -j9 && make install
        '''.format(**locals()))

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
        '--tag {}'.format(tag),
        '-f {}'.format(dockerfile_fpath),
        '.'
    ])
    print('docker_build_cli = {!r}'.format(docker_build_cli))
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
            file.write(tag)
        print(ub.color_text('\n--- SUCCESS ---', 'green'))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/hesaff/docker_build.py
    """
    main()
