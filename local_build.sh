#!/bin/bash

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



#### --- MATRIX --- ####
# The env part of travis.yml
## TODO: vary depending on platform
export MB_PYTHON_VERSION=3.6
export ENABLE_CONTRIB=0
export ENABLE_HEADLESS=1


#### --- BEFORE INSTALL --- ####
#set -e

git clone https://github.com/matthew-brett/multibuild.git

source multibuild/common_utils.sh

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then export ARCH_FLAGS=" "; fi

source multibuild/travis_steps.sh
# This sets -x
source travis_multibuild_customize.sh
echo $ENABLE_CONTRIB > contrib.enabled
echo $ENABLE_HEADLESS > headless.enabled


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
