#!/bin/bash
__heredoc__="""
Execute the multibuild.

This file is the entry point for a multibuild. It can either be run locally in
the root of the primary repo checkout, or it can be run via a CI server via
travis. The specific binary will (try) to target the users environment by
default. 

Note that this script uses the network to stage its dependencies.
"""

set -x
TEST_DEPENDS="numpy xdoctest ubelt"
CONFIG_PATH="multibuild_config.sh"
BDIST_PARAMS=""
USE_CCACHE=1
PLAT=${PLAT:$(arch)}
UNICODE_WIDTH=32
MB_PYTHON_VERSION=$(python -c "import sys; print('{}.{}'.format(*sys.version_info[0:2]))")

pip install xdoctest

source stage_multibuild.sh
set +x
set +e
echo "MB_PYTHON_VERSION = $MB_PYTHON_VERSION"
echo "PLAT = $PLAT"
echo "REPO_DIR = $REPO_DIR"
echo "_SOURCE_REPO = $_SOURCE_REPO"
echo "_STAGED_REPO = $_STAGED_REPO"

#### --- INSTALL --- ####

echo "--- BEGIN EXEC BUILD WHEEL ---"
set -e
set -x
cd $_STAGED_REPO
build_wheel $REPO_DIR $PLAT
set +x
echo "--- END EXEC BUILD WHEEL ---"
echo "REPO_DIR = $REPO_DIR"
cd $_SOURCE_REPO
