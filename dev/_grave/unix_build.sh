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
echo "[hesaff.unix_build] checking if build dir should be removed"
#python -c "import utool as ut; print('keeping build dir' if ut.get_argflag(('--fast', '--no-rmbuild')) else ut.delete('build'))" $@

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
#export LOCAL_PREFIX_CONDA=$HOME/anaconda3
export LOCAL_PREFIX_CONDA=$CONDA_PREFIX

if [[ "$VIRTUAL_ENV" == ""  ]] && [[ "$CONDA_PREFIX" == "" ]]; then
    export LOCAL_PREFIX=/usr/local
    export _SUDO="sudo"
else
    if [[ "$CONDA_PREFIX" == "" ]]; then
        export LOCAL_PREFIX=$(python -c "import sys; print(sys.prefix)")
    else
        export LOCAL_PREFIX=$LOCAL_PREFIX_CONDA
        echo 'If this fails run `conda install opencv` and then try again'
        # Ensure:
        # conda install opencv
    fi
    export _SUDO=""
fi


if [[ "$OSTYPE" == "msys"* ]]; then
    echo "INSTALL32=$INSTALL32"
    echo "HESAFF_INSTALL=$HESAFF_INSTALL"
    export INSTALL32="c:/Program Files (x86)"
    export OPENCV_DIR=$INSTALL32/OpenCV
else
    echo $LOCAL_PREFIX
    export OPENCV_DIR=$LOCAL_PREFIX/share/OpenCV
fi

if [[ "$OSTYPE" != "darwin"* ]]; then
    if [[ ! -d $OPENCV_DIR ]]; then
        { echo "FAILED OPENCV DIR DOES NOT EXIST" ; exit 1; }
    fi
fi

echo 'Configuring with cmake'
if [[ "$OSTYPE" == "msys"* ]]; then
    # WINDOWS
    echo "USE MINGW BUILD INSTEAD" ; exit 1
    cmake -G "MSYS Makefiles" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL32/Hesaff" \
        -DOpenCV_DIR=$OPENCV_DIR \
        ..
else
    # LINUX
    cmake -G "Unix Makefiles" \
        -DCMAKE_INSTALL_PREFIX=$LOCAL_PREFIX \
        -DOpenCV_DIR=$OPENCV_DIR \
        -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}" \
        -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}" \
        -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS}" \
        -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS}" \
        ..
fi

export CMAKE_EXITCODE=$?
if [[ $CMAKE_EXITCODE != 0 ]]; then
    { echo "FAILED HESAFF BUILD - CMake Step" ; exit 1; }
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
    { echo "FAILED HESAFF BUILD - Make step" ; exit 1; }
fi
cd ..
