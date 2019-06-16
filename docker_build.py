#!/usr/bin/env python
"""
References:
    https://github.com/skvark/opencv-python
"""
from __future__ import absolute_import, division, print_function
import ubelt as ub
import setup
import sys
import os
from os.path import join, exists


def setup_staging():
    """
    # TODO: make robust

    OPENCV_VERSION="4.1.0"
    cd ~/code/hesaff/docker/staging
    wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip

    unzip ${OPENCV_VERSION}.zip
    """
    pass


def stage_self(ROOT, staging_dpath):
    import shutil
    # stage the important files in this repo
    dist_paths = [
        'pyhesaff', 'src', 'CMakeLists.txt', 'setup.py', 'run_doctests.sh',
        'CMake', 'run_tests.py'
    ]
    from os.path import isfile, exists

    def copy3(src, dst):
        if exists(dst) and isfile(dst):
            os.unlink(dst)
        shutil.copy2(src, dst)

    mirror_dpath = ub.ensuredir((staging_dpath, 'hesaff'))

    copy_function = shutil.copy2
    copy_function = copy3
    print('======')
    for pname in dist_paths:
        src = join(ROOT, pname)
        dst = join(mirror_dpath, pname)
        print('src={!r}, dst={!r}'.format(src, dst))

        if os.path.isdir(pname):
            ub.delete(dst)
            shutil.copytree(src, dst, copy_function=copy_function)
        else:
            copy_function(src, dst)
    print('======')


def main():
    import os
    ROOT = join(os.getcwd())
    ROOT = ub.expandpath('~/code/hesaff')
    os.chdir(ROOT)

    VERSION = setup.version
    PY_VER = sys.version_info.major
    NAME = 'pyhesaff'
    tag = '{}-{}-py{}'.format(NAME, VERSION, PY_VER)

    # context_dpath = ub.ensuredir((ROOT, 'docker/context'))
    staging_dpath = ub.ensuredir((ROOT, 'docker/staging'))

    # Prestage the multibuild repo
    if not exists(join(staging_dpath, 'multibuild')):
        # FIXME: make robust in the case this fails
        info = ub.cmd('git clone https://github.com/matthew-brett/multibuild.git', cwd=staging_dpath, verbose=3)

    if not exists(join(staging_dpath, 'opencv')):
        # FIXME: make robust in the case this fails
        opencv_version = '4.1.0'
        fpath = ub.grabdata('https://github.com/opencv/opencv/archive/{}.zip'.format(opencv_version), verbose=1)
        ub.cmd('ln -s {} .'.format(fpath), cwd=staging_dpath, verbose=3)
        ub.cmd('unzip {}'.format(fpath), cwd=staging_dpath, verbose=3)
        import shutil
        shutil.move(join(staging_dpath, 'opencv-' + opencv_version), join(staging_dpath, 'opencv'))

    stage_self(ROOT, staging_dpath)

    dockerfile_fpath = join(ROOT, 'Dockerfile')
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
        ENV PYTHON_EXE=/opt/python/cp36-cp36m/python
        ENV MULTIBUILD_DIR=/root/code/multibuild
        ENV HOME=/root
        # params to bdist_wheel. used to set osx build target.
        ENV TEST_DEPENDS="numpy==1.11.1"
        ENV BDIST_PARAMS=""
        ENV USE_CCACHE=1
        ENV PLAT=x86_64
        ENV UNICODE_WIDTH=32

        # -e BUILD_COMMANDS="$build_cmds" \
        # -e PYTHON_VERSION="$MB_PYTHON_VERSION" \
        # -e UNICODE_WIDTH="$UNICODE_WIDTH" \
        # -e BUILD_COMMIT="$BUILD_COMMIT" \
        # -e CONFIG_PATH="$CONFIG_PATH" \
        # -e ENV_VARS_PATH="$ENV_VARS_PATH" \
        # -e WHEEL_SDIR="$WHEEL_SDIR" \
        # -e MANYLINUX_URL="$MANYLINUX_URL" \
        # -e BUILD_DEPENDS="$BUILD_DEPENDS" \
        # -e USE_CCACHE="$USE_CCACHE" \
        # -e REPO_DIR="$repo_dir" \
        # -e PLAT="$PLAT" \

        # These are defined in the parent image
        # ENV JPEG_INCLUDE_DIR=/opt/libjpeg-turbo/include
        # ENV JPEG_LIBRARY=/opt/libjpeg-turbo/lib64/libjpeg.a

        RUN mkdir -p /io
        WORKDIR /root

        # Setup code / scripts
        COPY docker/staging/multibuild /root/code/multibuild
        # Hack to fix issue
        RUN find $MULTIBUILD_DIR -iname "*.sh" -type f -exec sed -i 's/gh-clone/gh_clone/g' {} +

        # Copy bash configs (mirrors the environs)
        COPY docker/config.sh /root/config.sh
        COPY docker/bashrc.sh /root/.bashrc

        # Setup a virtualenv
        RUN source /root/.bashrc && \
            $PYTHON_EXE -m pip install --upgrade pip && \
            $PYTHON_EXE -m pip install virtualenv && \
            $PYTHON_EXE -m virtualenv --python=$PYTHON_EXE $HOME/venv

        # Install packages in virtual environment
        RUN source /root/.bashrc && \
            pip install cmake ninja scikit-build wheel numpy

        # This is very different for different operating systems
        # https://github.com/skvark/opencv-python/blob/master/setup.py
        COPY docker/staging/opencv /root/code/opencv
        RUN source /root/.bashrc && \
            source code/multibuild/common_utils.sh && \
            source code/multibuild/travis_linux_steps.sh && \
            mkdir -p /root/code/opencv/build && \
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
                   -DJPEG_INCLUDE_DIR=/opt/libjpeg-turbo/include \
                   -DJPEG_LIBRARY=/opt/libjpeg-turbo/lib64/libjpeg.a \
                /root/code/opencv

       # Note: there is no need to compile the above with python
       # -DPYTHON3_EXECUTABLE=$PYTHON_EXE \
       # -DBUILD_opencv_python3=ON \
       # -DOPENCV_PYTHON3_INSTALL_PATH=python \

        RUN source /root/.bashrc && \
            source code/multibuild/common_utils.sh && \
            source code/multibuild/travis_linux_steps.sh && \
            cd /root/code/opencv/build && \
            make -j9 && make install

        COPY docker/staging/hesaff /root/code/hesaff

        WORKDIR /root/code/hesaff
        RUN source /root/.bashrc && \
            python -m pip install -r requirements.txt

        # Use skbuild to build hesaff
        RUN source /root/.bashrc && \
            CMAKE_FIND_LIBRARY_SUFFIXES=".a;.so" python setup.py build_ext --inplace

        RUN source /root/.bashrc && \
            pip install xdoctest
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

        VMNT_DIR={ROOT}/{NAME}-docker/vmnt
        mkdir -p VMNT_DIR
        TAG={tag}

        # Test that we can get a bash terminal
        docker run -v $VMNT_DIR:/root/vmnt -it {tag} bash

        # Move deployment to the vmnt directory
        docker run -v $VMNT_DIR:/root/vmnt -it {tag} bash -c 'cd /root/code/hesaff && python3 -m xdoctest pyhesaff'

        # Run system tests
        docker run -v $VMNT_DIR:/root/vmnt -it {tag} bash -c 'cd /root/code/hesaff && python3 run_doctests.sh'

        # Inside bash test that we can fit a new model
        python -m pyhessaff demo

        mkdir -p ${ROOT}/{NAME}-docker/dist
        docker save -o ${ROOT}/{NAME}-docker/dist/{tag}.docker.tar {tag}
        ''').format(NAME=NAME, ROOT=ROOT, tag=tag), 'bash'))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/hesaff/docker_build.py
    """
    main()
