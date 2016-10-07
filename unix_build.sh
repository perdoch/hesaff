#!/bin/bash

# +==================================================
# SIMPLE WAY OF EXECUTING MULTILINE PYTHON FROM BASH
# +--------------------------------------------------
# Creates custom file descriptor that runs the script
# References: http://superuser.com/questions/607367/raw-multiline-string-in-bash
#exec 42<<'__PYSCRIPT__'
#import utool as ut;

#if not ut.get_argflag('--no-rmbuild'):
#    print('deleting build dir')
#    ut.delete('build')
#else:
#    print('keeping build dir')
#__PYSCRIPT__
#python /dev/fd/42 $@
# L_________________________________________________
export FAILCMD='{ echo "FAILED HESAFF BUILD" ; exit 1; }'
echo "[hesaff.unix_build] checking if build dir should be removed"
python2.7 -c "import utool as ut; print('keeping build dir' if ut.get_argflag(('--fast', '--no-rmbuild')) else ut.delete('build'))" $@

#################################
echo 'Removing old build'
rm -rf build
rm -rf CMakeFiles
rm -rf CMakeCache.txt
rm -rf cmake_install.cmake
#################################
echo 'Creating new build'
mkdir build
cd build
#################################

export PYEXE=$(which python2.7)
if [[ "$VIRTUAL_ENV" == ""  ]]; then
    export LOCAL_PREFIX=/usr/local
    export _SUDO="sudo"
else
    export LOCAL_PREFIX=$($PYEXE -c "import sys; print(sys.prefix)")/local
    export _SUDO=""
fi

echo 'Configuring with cmake'
if [[ '$OSTYPE' == 'darwin'* ]]; then
    export CONFIG="-DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_C_COMPILER=clang2 -DCMAKE_CXX_COMPILER=clang2++ -DCMAKE_INSTALL_PREFIX=$LOCAL_PREFIX -DOpenCV_DIR=$LOCAL_PREFIX/share/OpenCV"
    cmake $CONFIG -G 'Unix Makefiles' ..
elif [[ "$OSTYPE" == "msys"* ]]; then
    # WINDOWS
    echo "USE MINGW BUILD INSTEAD" ; exit 1
    export INSTALL32="c:/Program Files (x86)"
    export HESAFF_INSTALL="$INSTALL32/Hesaff"
    echo "INSTALL32=$INSTALL32"
    echo "HESAFF_INSTALL=$HESAFF_INSTALL"

    cmake -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX="$HESAFF_INSTALL" -DOpenCV_DIR="$INSTALL32/OpenCV" $COMMONFLAGS ..
else
    # LINUX
    cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=$LOCAL_PREFIX -DOpenCV_DIR=$LOCAL_PREFIX/share/OpenCV $COMMONFLAGS ..
fi

export CMAKE_EXITCODE=$?
if [[ $CMAKE_EXITCODE != 0 ]]; then
    $FAILCMD
fi

if [[ "$OSTYPE" == "msys"* ]]; then
    make
else
    export NCPUS=$(grep -c ^processor /proc/cpuinfo)
    #make
    make -j$NCPUS || make
    #make -j$NCPUS VERBOSE=1
    #make -j$NCPUS
fi
export MAKE_EXITCODE=$?
echo "MAKE_EXITCODE=$MAKE_EXITCODE"

if [[ $MAKE_EXITCODE == 0 ]]; then
    #make VERBOSE=1
    cp -v libhesaff* ../pyhesaff
else
    $FAILCMD
fi
cd ..
