#!/bin/bash
__heredoc__="""
Execute the multibuild.

This file is the entry point for a multibuild. It can either be run locally in
the root of the primary repo checkout, or it can be run via a CI server via
travis. The specific binary will (try) to target the users environment by
default. 

Note that this script uses the network to stage its dependencies.
"""

#TEST_DEPENDS="numpy xdoctest ubelt"
#CONFIG_PATH="multibuild_config.sh"
#BDIST_PARAMS=""
#USE_CCACHE=1
#PLAT=${PLAT:$(arch)}
#UNICODE_WIDTH=32
#MB_PYTHON_VERSION=$(python -c "import sys; print('{}.{}'.format(*sys.version_info[0:2]))")
#pip install ubelt xdoctest

# All the interesting stuff lives here
source factory/stage_multibuild.sh

#echo "MB_PYTHON_VERSION=$MB_PYTHON_VERSION"
#echo "DOCKER_IMAGE=$DOCKER_IMAGE"
#echo "PLAT=$PLAT"
#echo "REPO_DIR=$REPO_DIR"
#echo "_SOURCE_REPO=$_SOURCE_REPO"
#echo "_STAGED_REPO=$_STAGED_REPO"
#### --- INSTALL --- ####

echo "--- BEGIN EXEC BUILD WHEEL ---"
#cd $_STAGED_REPO
build_wheel $REPO_DIR $PLAT
#mkdir -p $_SOURCE_REPO/dist
#mkdir -p $_SOURCE_REPO/wheelhouse
#cp $_STAGED_REPO/dist/*.whl $_SOURCE_REPO/dist
#cp $_STAGED_REPO/dist/*.whl $_SOURCE_REPO/wheelhouse
echo "--- END EXEC BUILD WHEEL ---"

## Build and package
#set -x
##source run_multibuild.sh
#cd $_STAGED_REPO
#echo "_STAGED_REPO = $_STAGED_REPO"
#echo "REPO_DIR = $REPO_DIR"
#build_wheel $REPO_DIR $PLAT
#ls $_STAGED_REPO/wheelhouse
#cp $_STAGED_REPO/wheelhouse $_SOURCE_REPO/wheelhouse
#echo "_SOURCE_REPO = $_SOURCE_REPO"
#cd $_SOURCE_REPO
#set +x
#set +e
