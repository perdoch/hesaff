#!/usr/bin/env python
"""
References:
    https://github.com/skvark/opencv-python
"""
from __future__ import absolute_import, division, print_function
import ubelt as ub
from os.path import join


def main():
    import shutil  # NOQA
    import setup
    import sys
    import os

    ROOT = os.getcwd()
    VERSION = setup.version
    PY_VER = sys.version_info.major
    NAME = 'pyhesaff'
    tag = '{}-{}-py{}'.format(NAME, VERSION, PY_VER)
    dockerfile_relpath = join(ROOT, 'Dockerfile')
    context_dpath = ub.expandpath(join(ROOT, 'pyhesaff-docker/context'))
    ub.ensuredir(context_dpath)

    dist_paths = [
        'Dockerfile',
        'pyhesaff',
        'src',
        'CMakeLists.txt',
        'setup.py',
        'run_doctests.sh',
    ]
    from os.path import isfile, exists

    def copy3(src, dst):
        if exists(dst) and isfile(dst):
            os.unlink(dst)
        shutil.copy2(src, dst)

    copy_function = shutil.copy2
    copy_function = copy3
    for pname in dist_paths:
        src = join(ROOT, pname)
        dst = join(context_dpath, pname)
        print('======')
        print('src = {!r}'.format(src))
        print('dst = {!r}'.format(dst))

        if os.path.isdir(pname):
            ub.delete(dst)
            shutil.copytree(src, dst, copy_function=copy_function)
        else:
            copy_function(src, dst)

    docker_build_cli = ' '.join([
        'docker', 'build',
        # '--build-arg PY_VER={}'.format(PY_VER),
        '--tag {}'.format(tag),
        '-f {}'.format(dockerfile_relpath),
        '.'
    ])
    print('docker_build_cli = {!r}'.format(docker_build_cli))
    info = ub.cmd(docker_build_cli, verbose=3, shell=True)

    if info['ret'] != 0:
        print('Failed command:')
        print(info['command'])
        print(info['err'])
        print('NOTE: sometimes reruning the command manually works')
        raise Exception('Building docker failed with exit code {}'.format(info['ret']))

    # print(ub.highlight_code(ub.codeblock(
    print(ub.codeblock(
        r'''
        # Finished creating the docker image.
        # To test / export you can do something like this:

        VMNT_DIR={ROOT}/{NAME}-docker/vmnt
        mkdir -p VMNT_DIR
        TAG={tag}

        # Move deployment to the vmnt directory
        docker run -v $VMNT_DIR:/root/vmnt -it {tag} bash -c 'cd /root/code/hesaff && python3 -m xdoctest pyhesaff'

        # Run system tests
        docker run -v $VMNT_DIR:/root/vmnt -it {tag} bash -c 'cd /root/code/hesaff && python3 run_doctests.sh'

        # Test that we can get a bash terminal
        docker run -v $VMNT_DIR:/root/vmnt -it {tag} bash

        # Inside bash test that we can fit a new model
        python -m pyhessaff demo

        mkdir -p ${ROOT}/{NAME}-docker/dist
        docker save -o ${ROOT}/{NAME}-docker/dist/{tag}.docker.tar {tag}
        ''').format(NAME=NAME, ROOT=ROOT, tag=tag))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/hesaff/docker_build.py
    """
    main()
