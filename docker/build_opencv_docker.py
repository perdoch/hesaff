#!/usr/bin/env python
"""
References:
    https://github.com/skvark/opencv-python
"""
from __future__ import absolute_import, division, print_function
import ubelt as ub
import sys
import os
from os.path import join, exists


def main():
    ROOT = join(os.getcwd())
    ROOT = ub.expandpath('~/code/hesaff')
    os.chdir(ROOT)

    BASE = 'manylinux1_x86_64'

    PY_VER = sys.version_info.major
    tag = '{}-opencv-py{}'.format(BASE, PY_VER)

    # context_dpath = ub.ensuredir((ROOT, 'docker/context'))
    staging_dpath = ub.ensuredir((ROOT, 'docker/staging'))

    if not exists(join(staging_dpath, 'opencv')):
        # FIXME: make robust in the case this fails
        opencv_version = '4.1.0'
        fpath = ub.grabdata('https://github.com/opencv/opencv/archive/{}.zip'.format(opencv_version), verbose=1)
        ub.cmd('ln -s {} .'.format(fpath), cwd=staging_dpath, verbose=3)
        ub.cmd('unzip {}'.format(fpath), cwd=staging_dpath, verbose=3)
        import shutil
        shutil.move(join(staging_dpath, 'opencv-' + opencv_version), join(staging_dpath, 'opencv'))

    dockerfile_fpath = join(ROOT, 'docker/Dockerfile')
    # This docker code is very specific for building linux binaries.
    # We will need to do a bit of refactoring to handle OSX and windows.
    # But the goal is to get at least one OS working end-to-end.
    docker_code = ub.codeblock(
        '''
        FROM quay.io/skvark/manylinux1_x86_64

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
        COPY docker/staging/opencv /root/code/opencv
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
        ''')

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
        print(ub.color_text('\n--- SUCCESS ---', 'green'))

    # print(ub.highlight_code(ub.codeblock(
    print(ub.highlight_code(ub.codeblock(
        r'''
        # Finished creating the docker image.
        # To test / export you can do something like this:

        # Test that we can get a bash terminal
        docker run -it {tag} bash
        ''').format(ROOT=ROOT, tag=tag), 'bash'))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/hesaff/docker_build.py
    """
    main()
