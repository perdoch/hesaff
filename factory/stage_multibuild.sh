__THIS_DIR=$(dirname "${BASH_SOURCE[0]}")
source $__THIS_DIR/factory_utils.sh


setup_staging_helper


echo "BASH_SOURCE = $BASH_SOURCE"
# Change directory into the staging copy and procede with the build
#echo "ATEMPTING TO CD"
#set +e

#cd "${_STAGED_REPO}"
set -e
REPO_DIR="${_STAGED_REPO}"
#REPO_DIR="."

if [[ "$TRAVIS_OS_NAME" == "osx" ]]; 
    then export ARCH_FLAGS=" "; 
fi

source multibuild/common_utils.sh
source multibuild/travis_steps.sh


# I have no idea what this does
#if [ -n "$IS_OSX" ]; then
#    osx_staging_helper
#fi

echo "=== END OF STAGE MULTIBUILD ==="
