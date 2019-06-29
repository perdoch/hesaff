if [ "$__STAGE_ONCE__" == "" ]; then
export __STAGE_ONCE__="TRUE"
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
if [[ "$MB_PYTHON_VERSION" = auto ]]; then
    MB_PYTHON_VERSION=$(python -c "import sys; print('{}.{}'.format(*sys.version_info[0:2]))")
fi

setup-staging(){
    REPO_NAME=hesaff
    _SOURCE_REPO=$(dirname "${BASH_SOURCE[0]}")
    _SOURCE_REPO=$(python -c "import os; print(os.path.realpath('$_SOURCE_REPO'))")
    echo "_SOURCE_REPO = $_SOURCE_REPO"

    # Hack for CI

    # if CI
    _STAGEING_DPATH=$_SOURCE_REPO
    _STAGED_REPO=$_SOURCE_REPO
    # else
    #_STAGEING_DPATH=$_SOURCE_REPO/_staging
    #_STAGED_REPO=$_STAGEING_DPATH/$REPO_NAME

    mkdir -p $_STAGEING_DPATH
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
    if [ ! -d multibuild ]; then
        git clone https://github.com/matthew-brett/multibuild.git multibuild
    fi
    #find multibuild -type f -exec sed -i.bak "s/ cd /#cd /g" {} \;
    sed -i "s/cd .repo_dir && .cmd .wheelhouse/\$cmd \$wheelhouse/g" multibuild/common_utils.sh
    #(cd multibuild && git diff common_utils.sh)
    #(cd multibuild && git checkout common_utils.sh)
    #cat multibuild/common_utils.sh | grep ".cmd.*wheel"
    
    #(cd $repo_dir && $cmd $wheelhouse)
    

    _USE_QUAY="True"
    if [ $_USE_QUAY = "True" ]; then
        # Assume that the query.io/erotemic/manylinux-opencv:{DOCKER_TAG} image exists
        echo "_STAGEING_DPATH = $_STAGEING_DPATH"
        python docker/build_opencv_docker.py --dpath=$_STAGEING_DPATH --no-exec
        DOCKER_TAG=$(cat $_STAGEING_DPATH/opencv-docker-tag.txt)
        DOCKER_IMAGE="quay.io/erotemic/manylinux-opencv:${DOCKER_TAG}"
        #docker pull $DOCKER_IMAGE
    else
        # Patch multibuild so we can start from a local docker image  
        find $_STAGED_REPO/multibuild -iname "*.sh" -type f -exec sed -i 's/ retry docker pull/ #retry docker pull/g' {} +
        # Ensure that the manylinux1_x86_64-opencv4.1.0-py3.6 docker image exists
        python docker/build_opencv_docker.py --dpath=$_STAGEING_DPATH --no-exec
        # Ensure that the manylinux1_x86_64-opencv4.1.0-py3.6 docker image exists
        DOCKER_TAG=$(cat $_STAGEING_DPATH/opencv-docker-tag.txt)
        DOCKER_IMAGE=$DOCKER_TAG
    fi
    echo "DOCKER_TAG = $DOCKER_TAG"
    echo "DOCKER_IMAGE = $DOCKER_IMAGE"
}

setup-staging


echo "BASH_SOURCE = $BASH_SOURCE"
# Change directory into the staging copy and procede with the build
#echo "ATEMPTING TO CD"
#set +e

#cd "${_STAGED_REPO}"
set -e
REPO_DIR="${_STAGED_REPO}"
#REPO_DIR="."

source multibuild/common_utils.sh
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then export ARCH_FLAGS=" "; fi
source multibuild/travis_steps.sh


# I have no idea what this does
if [ -n "$IS_OSX" ]; then
    echo "THIS IS OXS"
    TAPS="$(brew --repository)/Library/Taps"
    echo "TAPS = $TAPS"
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

#echo "_SOURCE_REPO = $_SOURCE_REPO"
#cd "${_SOURCE_REPO}"


echo "=== END OF STAGE MULTIBUILD ==="
fi
