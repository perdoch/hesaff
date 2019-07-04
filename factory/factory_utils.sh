#### --- GLOBAL --- ####
# env global for travis.yml
echo "=== START OF STAGE MULTIBUILD ==="

TEST_DEPENDS="numpy xdoctest ubelt"
CONFIG_PATH="multibuild_config.sh"
#BDIST_PARAMS=${BDIST_PARAMS:""}

USE_CCACHE=${USE_CCACHE:=1}
PLAT=${PLAT:=$(arch)}
UNICODE_WIDTH=${UNICODE_WIDTH:=32}  # TODO introspect
#python -c "import sysconfig, ubelt; print(ubelt.repr2(sysconfig.get_config_vars(), nl=1))" | grep -i width
#python -c "import sysconfig, ubelt; print(sysconfig.get_config_vars().get('Py_UNICODE_SIZE', 4) * 8)"
MB_PYTHON_VERSION=${MB_PYTHON_VERSION:=auto}

echo "MB_PYTHON_VERSION = $MB_PYTHON_VERSION"


setup_staging_helper(){
    if [[ "$MB_PYTHON_VERSION" = auto ]]; then
        echo "AUTOSETING"
        MB_PYTHON_VERSION=$(python -c "import sys; print('{}.{}'.format(*sys.version_info[0:2]))")
        echo "AUTOSET MB_PYTHON_VERSION = $MB_PYTHON_VERSION"
    fi

    REPO_NAME=hesaff
    #_SOURCE_REPO=$(dirname "${BASH_SOURCE[0]}")
    #_SOURCE_REPO=$(dirname $(dirname "${BASH_SOURCE[0]}"))
    _SOURCE_REPO=$(pwd)
    #_SOURCE_REPO=$(python -c "import os; print(os.path.realpath('$_SOURCE_REPO'))")
    echo "_SOURCE_REPO = $_SOURCE_REPO"

    # Hack for CI

    # if CI
    #_STAGEING_DPATH=$_SOURCE_REPO
    #_STAGED_REPO=$_SOURCE_REPO

    # NOTE: this path needs to be valid in docker and locally
    _STAGED_REPO="."
    _STAGEING_DPATH="."

    # else
    #_STAGEING_DPATH=$_SOURCE_REPO/_staging
    #_STAGED_REPO=$_STAGEING_DPATH/$REPO_NAME

    #mkdir -p $_STAGEING_DPATH
    #rm -rf $_STAGEING_DPATH/wheelhouse

    # Create a copy of this repo in the staging dir, but ignore build side effects
    # if CI
    #_EXCLUDE="'_staging','*.so','*.dylib','*.dll','_skbuild','*.egg-info','_dist','__pycache__','.git','dist*','build*','wheel*','dev','.git*','appveyor.yml','.travis.yml'"
    #rsync -avr --max-delete=0 --exclude={$_EXCLUDE} . $_STAGED_REPO 

    # Ensure multibuild exists in this copy of this repo
    #if [ ! -d $_STAGED_REPO/multibuild ]; then
    #    git clone https://github.com/matthew-brett/multibuild.git $_STAGED_REPO/multibuild
    #fi

    # HACK: clone to the local directory
    #if [ ! -d multibuild ]; then
    #    git clone https://github.com/matthew-brett/multibuild.git multibuild
    #fi
    #find multibuild -type f -exec sed -i.bak "s/ cd /#cd /g" {} \;
    #if [ -n "$IS_OSX" ]; then

    if [ "$TRAVIS_OS_NAME" = "osx" ]; then
        NEED_SED="False"
        _IS_LINUX="False"
    else
        _IS_LINUX="True"
    fi

    #if [ "$NEED_SED" = "True" ]; then
    #    #(cd multibuild && git checkout common_utils.sh)
    #    sed -i "s/cd .repo_dir && .cmd .wheelhouse/\$cmd \$wheelhouse/g" multibuild/common_utils.sh
    #fi
    #fi
    __comment__="""
    (cd multibuild && git diff common_utils.sh)
    (cd multibuild && git checkout common_utils.sh)
    cat multibuild/common_utils.sh | grep ".cmd.*wheel"
    (cd $repo_dir && $cmd $wheelhouse)
    """

    if [ $_IS_LINUX = "True" ]; then
        _USE_QUAY="True"
        if [ $_USE_QUAY = "True" ]; then
            # Assume that the query.io/erotemic/manylinux-opencv:{DOCKER_TAG} image exists
            echo "_STAGEING_DPATH = $_STAGEING_DPATH"
            python factory/build_opencv_docker.py --dpath=$_STAGEING_DPATH --no-exec
            DOCKER_TAG=$(cat $_STAGEING_DPATH/opencv-docker-tag.txt)
            DOCKER_IMAGE="quay.io/erotemic/manylinux-opencv:${DOCKER_TAG}"
            #docker pull $DOCKER_IMAGE
        else
            # Patch multibuild so we can start from a local docker image  
            find $_STAGED_REPO/multibuild -iname "*.sh" -type f -exec sed -i 's/ retry docker pull/ #retry docker pull/g' {} +
            # Ensure that the manylinux1_x86_64-opencv4.1.0-py3.6 docker image exists
            python factory/build_opencv_docker.py --dpath=$_STAGEING_DPATH --no-exec
            # Ensure that the manylinux1_x86_64-opencv4.1.0-py3.6 docker image exists
            DOCKER_TAG=$(cat $_STAGEING_DPATH/opencv-docker-tag.txt)
            DOCKER_IMAGE=$DOCKER_TAG
        fi
        echo "DOCKER_TAG = $DOCKER_TAG"
        echo "DOCKER_IMAGE = $DOCKER_IMAGE"
    fi
}

if [ -n "$IS_OSX" ]; then
    source $__THIS_DIR/osx_utils.sh
fi
