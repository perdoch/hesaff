#!/bin/bash
#cd ~/code/hesaff

export FAILCMD='{ echo "FAILED HESAFF BUILD" ; exit 1; }'

echo "[hesaff.unix_build] checking if build dir should be removed"
python2.7 -c "import utool as ut; print('keeping build dir' if ut.get_argflag(('--fast', '--no-rmbuild')) else ut.delete('build'))" $@

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

mkdir build
cd build

echo "$OSTYPE"

export PYEXE=$(which python2.7)
if [[ "$VIRTUAL_ENV" == ""  ]]; then
    export LOCAL_PREFIX=/usr/local
    export _SUDO="sudo"
else
    export LOCAL_PREFIX=$($PYEXE -c "import sys; print(sys.prefix)")/local
    export _SUDO=""
fi

export COMMONFLAGS="$COMMONFLAGS -DCMAKE_BUILD_TYPE=Release"
#export COMMONFLAGS="$COMMONFLAGS -D CMAKE_BUILD_TYPE=Debug"
#export COMMONFLAGS="$COMMONFLAGS -D CMAKE_VERBOSE_MAKEFILE=On"
#export COMMONFLAGS="$COMMONFLAGS -D ENABLE_GPROF=On"

echo "COMMONFLAGS=$COMMONFLAGS"

if [[ "$OSTYPE" == "darwin"* ]]; then
    # MAC
    cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=$LOCAL_PREFIX -DOpenCV_DIR=$LOCAL_PREFIX/share/OpenCV -DCMAKE_OSX_ARCHITECTURES=x86_64  $COMMONFLAGS ..
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
    make -j$NCPUS
    #make -j$NCPUS VERBOSE=1
    #make -j$NCPUS
fi
export MAKE_EXITCODE=$?
echo "MAKE_EXITCODE=$MAKE_EXITCODE"

if [[ $MAKE_EXITCODE == 0 ]]; then
    #make VERBOSE=1
    install_name_tool -change libiomp5.dylib ~/code/libomp_oss/exports/mac_32e/lib.thin/libiomp5.dylib lib*
    cp -v libhesaff* ../pyhesaff
else
    $FAILCMD
fi
cd ..
