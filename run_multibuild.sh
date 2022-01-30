#!/bin/bash
__heredoc__="""
DEPRECATED FOR CIBUILDWHEEL

Execute the multibuild.

This file is the entry point for a multibuild. It can either be run locally in
the root of the primary repo checkout, or it can be run via a CI server via
travis. The specific binary will (try) to target the users environment by
default. 

Note that this script uses the network to stage its dependencies.

New Way:

  CIBW_SKIP='pp*' cibuildwheel --config-file pyproject.toml --platform linux --arch x86_64


"""

# All the interesting stuff lives here
# from dev/stage_multibuild.sh import build_wheel
source dev/stage_multibuild.sh

echo "--- BEGIN EXEC BUILD WHEEL ---"
build_wheel $REPO_DIR $PLAT
echo "--- END EXEC BUILD WHEEL ---"
