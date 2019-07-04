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
      "ffmpeg_opencv 10" \
  )
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

function pre_build {
  echo "Starting pre-build"
  set -e -o pipefail

  if [ -n "$IS_OSX" ]; then
      prebuild_osx_brew_stuff
  else
    echo "Running for linux"
  fi
  #qmake -query

  PYTHON=python$PYTHON_VERSION
  $PYTHON -m pip install pip  -U
  if [ -n "$IS_OSX" ]; then
    echo "skip pip prebuild"
  else
    echo "Running for linux"
    $PYTHON -m pip install numpy scikit-build ubelt cmake ninja -U
  fi
}

function run_tests {
    # Runs tests on installed distribution from an empty directory
    echo "Run tests..."
    echo "PWD = $PWD"

    echo "PYTHON = $PYTHON"
    echo "PYTHON_VERSION = $PYTHON_VERSION"
    PYTHON=python$PYTHON_VERSION

    $PYTHON -m pip install pip  -U
    $PYTHON -m pip install git+https://github.com/Erotemic/fxdoctest.git@master
    $PYTHON -m xdoctest pyhesaff list
    echo "TODO: actually run tests"

    if [ -n "$IS_OSX" ]; then
      echo "Running for OS X"
      cd ../tests/
    else
      echo "Running for linux"
      cd /io/tests/
    fi

    test_wheels
}

function test_wheels {
    PYTHON=python$PYTHON_VERSION

    echo "Starting tests..."

    #Test package
    echo "TODO: looks like multibuild doesnt like xdoctest"
    #$PYTHON -m xdoctest pyhesaff
    #$PYTHON -m unittest test
    #$PYTHON -m xdoctest pyhesaff
}

export PS4='+(${BASH_SOURCE}:${LINENO}): ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
set -x
