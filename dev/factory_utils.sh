#### --- GLOBAL --- ####
# env global for travis.yml
echo "=== START OF STAGE MULTIBUILD ==="

TEST_DEPENDS="numpy ubelt"
BUILD_DEPENDS="numpy ubelt scikit-build ninja cmake"
#$PYTHON -m pip install git+https://github.com/Erotemic/xdoctest.git@master
CONFIG_PATH="multibuild_config.sh"
#BDIST_PARAMS=${BDIST_PARAMS:""}

USE_CCACHE=${USE_CCACHE:=1}
PLAT=${PLAT:=$(arch)}
UNICODE_WIDTH=${UNICODE_WIDTH:=32}  # TODO introspect
#python -c "import sysconfig, ubelt; print(ubelt.repr2(sysconfig.get_config_vars(), nl=1))" | grep -i width
#python -c "import sysconfig, ubelt; print(sysconfig.get_config_vars().get('Py_UNICODE_SIZE', 4) * 8)"
MB_PYTHON_VERSION=${MB_PYTHON_VERSION:=auto}
REPO_NAME=hesaff

_SOURCE_REPO=$(pwd)
_STAGED_REPO="."
_STAGEING_DPATH="."

# Hack in specific travis variables for local builds
if [ "$TRAVIS_OS_NAME" = "" ]; then 
    if [[ "$OSTYPE" == "darwin"* ]]; then 
        TRAVIS_OS_NAME="osx"
        echo "TRAVIS_OS_NAME = $TRAVIS_OS_NAME"
    elif [[ "$OSTYPE" == "linux"* ]]; then 
        TRAVIS_OS_NAME="linux"
        echo "TRAVIS_OS_NAME = $TRAVIS_OS_NAME"
    else
        TRAVIS_OS_NAME="UNKNOWN"
        echo "TRAVIS_OS_NAME = $TRAVIS_OS_NAME"
    fi
    if [ "$TRAVIS_BUILD_STAGE_NAME" = "" ]; then
        TRAVIS_BUILD_STAGE_NAME="final"
    fi
    if [ "$TRAVIS_TIMER_START_TIME" = "" ]; then 
        TRAVIS_TIMER_START_TIME=10000000000
    fi
fi


if [[ "$OSTYPE" == "darwin*" ]]; then 
    export ARCH_FLAGS=" "
fi

echo "
TRAVIS_OS_NAME = $TRAVIS_OS_NAME
TRAVIS_TIMER_START_TIME = $TRAVIS_TIMER_START_TIME
TRAVIS_BUILD_STAGE_NAME = $TRAVIS_BUILD_STAGE_NAME

ARCH_FLAGS = $ARCH_FLAGS

MB_PYTHON_VERSION = $MB_PYTHON_VERSION
REPO_NAME = $REPO_NAME
_SOURCE_REPO = $_SOURCE_REPO
_STAGED_REPO = $_STAGED_REPO
_STAGEING_DPATH = $_STAGEING_DPATH
"




setup_staging_helper(){
    if [[ "$MB_PYTHON_VERSION" = auto ]]; then
        echo "AUTOSETING"
        MB_PYTHON_VERSION=$(python -c "import sys; print('{}.{}'.format(*sys.version_info[0:2]))")
        echo "AUTOSET MB_PYTHON_VERSION = $MB_PYTHON_VERSION"
    fi

    # Hack for CI
    # NOTE: this path needs to be valid in docker and locally
    echo "
    MB_PYTHON_VERSION = $MB_PYTHON_VERSION
    _SOURCE_REPO = $_SOURCE_REPO
    _STAGED_REPO = $_STAGED_REPO
    _STAGEING_DPATH = $_STAGEING_DPATH
    "

    __comment__='''
    (cd multibuild && git diff common_utils.sh)
    (cd multibuild && git checkout common_utils.sh)
    cat multibuild/common_utils.sh | grep ".cmd.*wheel"
    (cd $repo_dir && $cmd $wheelhouse)
    '''

    if [[ "$OSTYPE" == "linux"* ]]; then
        _USE_QUAY="True"
        if [ $_USE_QUAY = "True" ]; then
            # Assume that the query.io/erotemic/manylinux-opencv:{DOCKER_TAG} image exists
            #python dev/build_opencv_docker.py --dpath=$_STAGEING_DPATH --no-exec
            #DOCKER_TAG=$(cat $_STAGEING_DPATH/opencv-docker-tag.txt)

            #DOCKER_TAG="x86_64-opencv4.1.0-v4"
            #PLAT
            DOCKER_TAG="${PLAT}-opencv4.1.0-v4"
            DOCKER_IMAGE="quay.io/erotemic/manylinux-for:${DOCKER_TAG}"
            
            #DOCKER_IMAGE="quay.io/erotemic/manylinux-opencv:${DOCKER_TAG}"
            #docker pull $DOCKER_IMAGE
        else
            # OLD AND DEPRECATED
            # Patch multibuild so we can start from a local docker image  
            find $_STAGED_REPO/multibuild -iname "*.sh" -type f -exec sed -i 's/ retry docker pull/ #retry docker pull/g' {} +
            # Ensure that the manylinux1_x86_64-opencv4.1.0-py3.6 docker image exists
            python dev/build_opencv_docker.py --dpath=$_STAGEING_DPATH --no-exec
            # Ensure that the manylinux1_x86_64-opencv4.1.0-py3.6 docker image exists
            DOCKER_TAG=$(cat $_STAGEING_DPATH/opencv-docker-tag.txt)
            DOCKER_IMAGE=$DOCKER_TAG
        fi
        echo "DOCKER_TAG = $DOCKER_TAG"
        echo "DOCKER_IMAGE = $DOCKER_IMAGE"
    fi
}


osx_staging_helper_clean_brew_cache(){
    echo "Executing OSX staging helper"
    TAPS="$(brew --repository)/Library/Taps"
    echo "TAPS = $TAPS"
    # Note: The tap command allows Homebrew to tap into another repository of formulae
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
    echo "Finished OSX staging helper"
}
