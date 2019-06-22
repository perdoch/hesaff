__heredoc__="""
This is the bashrc for the docker container's root user.
"""
export ENABLE_CONTRIB=0
export ENABLE_HEADLESS=1
export MB_PYTHON_VERSION=3.6
export PYTHON_VERSION=$MB_PYTHON_VERSION
export MULTIBUILD_DIR=/root/code/multibuild
#export BUILD_COMMANDS="echo 'this is a dummy build command'"

export TEST_DEPENDS="numpy==1.11.1"
export BDIST_PARAMS=""
export USE_CCACHE=1
export PLAT=x86_64
export UNICODE_WIDTH=32
export JPEG_INCLUDE_DIR=/opt/libjpeg-turbo/include
export JPEG_LIBRARY=/opt/libjpeg-turbo/lib64/libjpeg.a

export CONFIG_PATH=$HOME/config.sh

source $CONFIG_PATH

# HACK
source $MULTIBUILD_DIR/manylinux_utils.sh
source $MULTIBUILD_DIR/configure_build.sh
source $MULTIBUILD_DIR/library_builders.sh

export PYTHON_ROOT=$(cpython_path $PYTHON_VERSION)
export PYTHON_EXE=$PYTHON_ROOT/bin/python

#echo "PYTHON_EXE = $PYTHON_EXE"
#export PYTHONPATH=/opt/python/cp36-cp36m/lib/python3.6/site-packages/:$PYTHONPATH
#export PATH=/opt/python/cp36-cp36m/bin:$PATH
#export LD_LIBRARY_PATH=/opt/python/cp36-cp36m/lib:$LD_LIBRARY_PATH
#export PYTHON_EXE=/opt/python/cp36-cp36m/bin/python
#export PATH=$PYTHON_ROOT/bin:$PATH
#export LD_LIBRARY_PATH=$PYTHON_ROOT/lib:$LD_LIBRARY_PATH
#echo "PATH = $PATH"
#echo $($PYTHON_EXE --version)

if [ -d "$HOME/venv" ]; then 
    source $HOME/venv/bin/activate
fi

# Don't print out what we are doing
set +x
# Don't exit docker run on a failure
set +e
