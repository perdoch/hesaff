#export PS4='+(${BASH_SOURCE}:${LINENO}): ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
#set -x
source ./utils.sh

export MB_PYTHON_VERSION=3.6
export ENABLE_CONTRIB=1
export ENABLE_HEADLESS=0
export PYTHON_VERSION=$MB_PYTHON_VERSION
export BUILD_COMMANDS="echo 'hi'"
export MULTIBUILD_DIR=/root/code/multibuild

export CONFIG_PATH=dummy_config.sh
echo "echo 'sourcing my config_path'" >> $CONFIG_PATH

echo "MB_PYTHON_VERSION = $MB_PYTHON_VERSION"
echo "BUILD_COMMANDS = $BUILD_COMMANDS"
echo "CONFIG_PATH = $CONFIG_PATH"

mkdir -p /io

# HACK
source $MULTIBUILD_DIR/manylinux_utils.sh
source $MULTIBUILD_DIR/configure_build.sh
source $MULTIBUILD_DIR/library_builders.sh

export PYTHON_ROOT=$(cpython_path $PYTHON_VERSION)
export PYTHON_EXE=$PYTHON_ROOT/bin/python
#echo "PYTHON_EXE = $PYTHON_EXE"
export PYTHONPATH=/opt/python/cp36-cp36m/lib/python3.6/site-packages/:$PYTHONPATH
export PATH=/opt/python/cp36-cp36m/bin:$PATH
export LD_LIBRARY_PATH=/opt/python/cp36-cp36m/lib:$LD_LIBRARY_PATH
export PYTHON_EXE=/opt/python/cp36-cp36m/bin/python
#echo $($PYTHON_EXE --version)

export PATH=$PYTHON_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$PYTHON_ROOT/lib:$LD_LIBRARY_PATH

echo "PATH = $PATH"
echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"


if [ -d "$HOME/venv" ]; then 
    source $HOME/venv/bin/activate
fi

# Don't exit docker run on a failure
set +x


__heredoc__="""

docker run -it build_hesaff bash
docker build -t build_hesaff .

docker build -t build_hesaff . && docker run -it build_hesaff bash

source /root/config.sh
source $MULTIBUILD_DIR/docker_build_wrap.sh 

cat $MULTIBUILD_DIR/docker_build_wrap.sh 
cat $MULTIBUILD_DIR/docker_build_wrap.sh 
"""
