#!/bin/bash
#Customize multibuild logic that is run after entering docker.
#Sourced by docker_build_wrap.sh and docker_test_wrap.sh .
#Runs in Docker, so only the vars passed to `docker run' exist.
#See multibuild/README.rst
echo "===  Loading MULTIBUILD config.sh  === "


if [ -n "$IS_OSX" ]; then
  echo "    > OSX environment "
  export MAKEFLAGS="-j$(sysctl -n hw.ncpu)"

  source factory/travis_osx_brew_cache.sh
  BREW_SLOW_BUILIDING_PACKAGES=$(printf '%s\n' \
      "cmake 15" \
      "opencv 10" \
  )
  # Junk: "ffmpeg_opencv 10" \
else
  echo "    > Linux environment "
  export MAKEFLAGS="-j$(grep -E '^processor[[:space:]]*:' /proc/cpuinfo | wc -l)"
fi


# To see build progress
function build_wheel {
    build_bdist_wheel $@
}

function bdist_wheel_cmd {
    # copied from multibuild's common_utils.sh
    # add osx deployment target so it doesnt default to 10.6
    echo "-- !!!!!!!!!!!!!!!!!!!!!!!!! --"
    echo "-- IN CUSTOM BUILD WHEEL CMD --"
    local abs_wheelhouse=$1


    echo "BDIST_PARAMS = $BDIST_PARAMS"
    #rm -rf _skbuild

    #python setup.py build $BDIST_PARAMS
    # HACK TO GET LIBS IN THE RIGHT PLACE  
    echo "-- RUN SETUP BUILD --"
    python setup.py build_ext --inplace $BDIST_PARAMS
    echo "-- <SETUP BDIST_WHEEL> --"
    python setup.py bdist_wheel $BDIST_PARAMS
    echo "-- </SETUP BDIST_WHEEL> --"
    cp dist/*.whl $abs_wheelhouse
    if [ -n "$USE_CCACHE" -a -z "$BREW_BOOTSTRAP_MODE" ]; then ccache -s; fi

    echo "-- FINISH CUSTOM BUILD WHEEL CMD --"
    echo "-- !!!!!!!!!!!!!!!!!!!!!!!!! --"
}


#function before_install {
#    # Uninstall oclint. See Travis-CI gh-8826
#    brew cask uninstall oclint || true    
#    export CC=clang
#    export CXX=clang++
#    get_macpython_environment $MB_PYTHON_VERSION venv
#    source venv/bin/activate
#    pip install --upgrade pip wheel
#}


function pre_build {
  echo "Starting pre-build"
  # The set -e option will cause a bash script to exit immediately when a command fails
  # The -o pipefail option relaxes this, and allows pipes to catch exceptions.
  # IE. only fail if the rightmost command in a piped expression fails.
  set -e -o pipefail

  if [ -n "$IS_OSX" ]; then
    echo "Running for OSX"
    
    local CACHE_STAGE; (echo "$TRAVIS_BUILD_STAGE_NAME" | grep -qiF "final") || CACHE_STAGE=1
    echo "CACHE_STAGE = $CACHE_STAGE"

    echo "START pre_build install in TRAVIS_BUILD_STAGE_NAME=$TRAVIS_BUILD_STAGE_NAME with CACHE_STAGE=$CACHE_STAGE"

    #after the cache stage, all bottles and Homebrew metadata should be already cached locally
    if [ -n "$CACHE_STAGE" ]; then
        brew update
        #generate_ffmpeg_formula
        brew_add_local_bottles  # NOQA
    fi

    #echo 'Installing QT4'
    #brew tap | grep -qxF cartr/qt4 || brew tap cartr/qt4
    #brew tap --list-pinned | grep -qxF cartr/qt4 || brew tap-pin cartr/qt4
    #if [ -n "$CACHE_STAGE" ]; then
    #    brew_install_and_cache_within_time_limit qt@4 || { [ $? -gt 1 ] && return 2 || return 0; }
    #else
    #    brew install qt@4
    #fi

    #echo 'Installing FFmpeg'
    #if [ -n "$CACHE_STAGE" ]; then
    #    brew_install_and_cache_within_time_limit ffmpeg_opencv || { [ $? -gt 1 ] && return 2 || return 0; }
    #else
    #    brew install ffmpeg_opencv
    #fi

    echo "START openv install in TRAVIS_BUILD_STAGE_NAME='$TRAVIS_BUILD_STAGE_NAME'"
    if [ -n "$CACHE_STAGE" ]; then
        brew_install_and_cache_within_time_limit opencv || { [ $? -gt 1 ] && return 2 || return 0; }
    else
        #brew install opencv 
        # Numpy causes install opencv to partially fail, catch this and fix numpy once its done
        #start_spinner
        brew install opencv || brew link --overwrite numpy 
        #stop_spinner
    fi
    echo "FINISH openv install in TRAVIS_BUILD_STAGE_NAME='$TRAVIS_BUILD_STAGE_NAME'"

    if [ -n "$CACHE_STAGE" ]; then
        echo "START BREW_GO_BOOTSTRAP_MODE 0 TRAVIS_BUILD_STAGE_NAME='$TRAVIS_BUILD_STAGE_NAME'"
        brew_go_bootstrap_mode 0
        return 0
    fi
    
    # Have to install macpython late to avoid conflict with Homebrew Python update
    # THIS IS AN INLINE VERSION OF OSX before_install
    # before_install
    # ----------
    # Uninstall oclint. See Travis-CI gh-8826
    brew cask uninstall oclint || true    
    export CC=clang
    export CXX=clang++
    get_macpython_environment $MB_PYTHON_VERSION venv
    echo $?
    source venv/bin/activate
    pip install --quiet --upgrade pip wheel
    # ----------
  else
    echo "Running for linux"
    pip install pip  -U
  fi
  #qmake -query

  if [ -n "$IS_OSX" ]; then
    echo "skip pip prebuild"

    # https://medium.com/@nuwanprabhath/installing-opencv-in-macos-high-sierra-for-python-3-89c79f0a246a
    # Probably need to install opencv before running install
    #brew install opencv 
    pip install --quiet scikit-build ninja cmake ubelt numpy
  else
    echo "Running for linux"
    pip install --quiet numpy scikit-build ubelt cmake ninja 
  fi
}

function run_tests {
    # Runs tests on installed distribution from an empty directory
    echo "Run tests..."
    echo "PWD = $PWD"

    echo "PYTHON = $PYTHON"
    echo "PYTHON_VERSION = $PYTHON_VERSION"
    PYTHON=python$PYTHON_VERSION

    #if [[ "$OSTYPE" == "linux"* ]]; then
    #  https://github.com/Erotemic/xdoctest/archive/master.zip
    #fi
    #  https://github.com/Erotemic/xdoctest/archive/master.zip
    #$PYTHON -m pip install git+https://github.com/Erotemic/xdoctest.git@master
    #pip install https://github.com/Erotemic/xdoctest/archive/master.zip
    pip install --quiet opencv-python 
    pip install --quiet xdoctest

    # Install opencv-python for a working cv2 module. 
    python -m xdoctest pyhesaff list

    # TODO: more tests
    set -e
    echo "Execute real doctests"
    python -m xdoctest pyhesaff 

    #ls
    #ls wheelhouse
    #ls /Users/travis/build/Erotemic/hesaff/venv/lib/python2.7/site-packages/pyhesaff/
    # contains libhesaff.macosx-10.12-x86_64-2.7.dylib

    if [ -n "$IS_OSX" ]; then
      echo "Running for OS X"
      cd ../tests/
    else
      echo "Running for linux"
      cd /io/tests/
    fi
}

export PS4='+(${BASH_SOURCE}:${LINENO}): ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
set -x
