    #_SOURCE_REPO=$(dirname "${BASH_SOURCE[0]}")
    #_SOURCE_REPO=$(dirname $(dirname "${BASH_SOURCE[0]}"))
    #_SOURCE_REPO=$(python -c "import os; print(os.path.realpath('$_SOURCE_REPO'))")
    

    # if CI
    #_STAGEING_DPATH=$_SOURCE_REPO
    #_STAGED_REPO=$_SOURCE_REPO

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

    #if [ "$TRAVIS_OS_NAME" = "osx" ]; then
    #if [[ "$OSTYPE" = "darwin"* ]]; then
    #    NEED_SED="False"
    #    _IS_LINUX="False"
    #else
    #    _IS_LINUX="True"
    #fi

    #if [ "$NEED_SED" = "True" ]; then
    #    #(cd multibuild && git checkout common_utils.sh)
    #    sed -i "s/cd .repo_dir && .cmd .wheelhouse/\$cmd \$wheelhouse/g" multibuild/common_utils.sh
    #fi
    #fi
