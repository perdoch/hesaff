#!/bin/bash
__heredoc__="""
"""

#### --- GLOBAL --- ####

# env global for travis.yml
#"PS4='+(${BASH_SOURCE}:${LINENO}): ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'"
# pip dependencies to _test_ your project
export TEST_DEPENDS="numpy xdoctest ubelt"
# params to bdist_wheel. used to set osx build target.
export BDIST_PARAMS=""
export USE_CCACHE=1
export PLAT=x86_64
export UNICODE_WIDTH=32
export CONFIG_PATH="travis_config.sh"


# TODO: PASS THESE IN VIA PARAMS


#### --- MATRIX --- ####
# The env part of travis.yml
## TODO: vary depending on platform
export MB_PYTHON_VERSION=3.6


#### --- BEFORE INSTALL --- ####
export PS4='+(${BASH_SOURCE}:${LINENO}): ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
set -e
set -x

setup-staging(){ 
    REPO_NAME=hesaff
    THIS_REPO_DIR=$(realpath $(dirname "${BASH_SOURCE[0]}"))
    STAGING_DPATH=$THIS_REPO_DIR/_staging
    REPO_DIR=$STAGING_DPATH/$REPO_NAME
    mkdir -p $STAGING_DPATH

    #echo "THIS_REPO_DIR = $THIS_REPO_DIR"
    #echo "REPO_DIR = $REPO_DIR"

    # Create a copy of this repo in the staging dir, but ignore build side effects
    _EXCLUDE="'_staging','*.so','*.dylib','*.dll','_skbuild','*.egg.*','_dist','__*'"
    bash -c "rsync -avrP --exclude={$_EXCLUDE} . $REPO_DIR"  # wrapped due to format issue in editor

    # Ensure multibuild exists in this copy of this repo
    if [ ! -d $REPO_DIR/multibuild ]; then
        git clone https://github.com/matthew-brett/multibuild.git $REPO_DIR/multibuild
    fi
    # Patch multibuild so we can start from a local docker image  
    find $REPO_DIR/multibuild -iname "*.sh" -type f -exec sed -i 's/ retry docker pull/ #retry docker pull/g' {} +

    # Ensure that the manylinux1_x86_64-opencv4.1.0-py3.6 docker image exists
    python docker/build_opencv_docker.py --dpath=$STAGING_DPATH

    DOCKER_IMAGE=$(cat $STAGING_DPATH/opencv-docker-tag.txt)
    echo "DOCKER_IMAGE = $DOCKER_IMAGE"
}

setup-staging


# Change directory into the staging copy and procede with the build
cd $REPO_DIR
source $REPO_DIR/multibuild/common_utils.sh
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then export ARCH_FLAGS=" "; fi
source $REPO_DIR/multibuild/travis_steps.sh


# I have no idea what this does
if [ -n "$IS_OSX" ]; then
    TAPS="$(brew --repository)/Library/Taps"
    if [ -e "$TAPS/caskroom/homebrew-cask" -a -e "$TAPS/homebrew/homebrew-cask" ]; then
        rm -rf "$TAPS/caskroom/homebrew-cask"
    fi
    find "$TAPS" -type d -name .git -exec \
            bash -xec '
                cd $(dirname '\''{}'\'') || echo "status: $?"
                git clean -fxd || echo "status: $?"
                sleep 1 || echo "status: $?"
                git status || echo "status: $?"' \; || echo "status: $?"

    brew_cache_cleanup
fi


#### --- INSTALL --- ####
build_wheel $REPO_DIR $PLAT
