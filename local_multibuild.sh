#!/bin/bash
__heredoc__="""
Execute the multibuild.

This file is the entry point for a multibuild. It can either be run locally in
the root of the primary repo checkout, or it can be run via a CI server via
travis. The specific binary will (try) to target the users environment by
default. 

Note that this script uses the network to stage its dependencies.
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
export CONFIG_PATH="multibuild_config.sh"

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
    _SOURCE_REPO=$(realpath $(dirname "${BASH_SOURCE[0]}"))
    _STAGEING_DPATH=$_SOURCE_REPO/_staging
    _STAGED_REPO=$_STAGEING_DPATH/$REPO_NAME
    mkdir -p $_STAGEING_DPATH

    #echo "_SOURCE_REPO = $_SOURCE_REPO"
    #echo "_STAGED_REPO = $_STAGED_REPO"

    # Create a copy of this repo in the staging dir, but ignore build side effects
    _EXCLUDE="'_staging','*.so','*.dylib','*.dll','_skbuild','*.egg.*','_dist','__pycache__','.git'"
    bash -c "rsync -avrP --exclude={$_EXCLUDE} . $_STAGED_REPO"  # wrapped due to format issue in editor

    # Ensure multibuild exists in this copy of this repo
    if [ ! -d $_STAGED_REPO/multibuild ]; then
        git clone https://github.com/matthew-brett/multibuild.git $_STAGED_REPO/multibuild
    fi
    # Patch multibuild so we can start from a local docker image  
    find $_STAGED_REPO/multibuild -iname "*.sh" -type f -exec sed -i 's/ retry docker pull/ #retry docker pull/g' {} +

    # Ensure that the manylinux1_x86_64-opencv4.1.0-py3.6 docker image exists
    python docker/build_opencv_docker.py --dpath=$_STAGEING_DPATH

    DOCKER_IMAGE=$(cat $_STAGEING_DPATH/opencv-docker-tag.txt)
    echo "DOCKER_IMAGE = $DOCKER_IMAGE"
}

setup-staging


echo "BASH_SOURCE = $BASH_SOURCE"
# Change directory into the staging copy and procede with the build
cd $_STAGED_REPO
REPO_DIR="."
mkdir -p wheelhouse

source multibuild/common_utils.sh
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then export ARCH_FLAGS=" "; fi
source multibuild/travis_steps.sh


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
