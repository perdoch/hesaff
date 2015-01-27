#!/bin/bash
#cd ~/code/hesaff
echo "[hesaff.unix_build] checking if build dir should be removed"

#RMBUILD=1
#for i in "$@"
#do
#case $i in --no-rmbuild)
    #RMBUILD=0
    #;;
#esac
#done
#if [[ "$RMBUILD" == "1" ]]; then
    #rm -rf build
#fi
python -c "import utool as ut; print('keeping build dir' if ut.get_argflag('--no-rmbuild') else ut.delete('build'))" $@

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

if [[ "$OSTYPE" == "darwin"* ]]; then
    cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 -G "Unix Makefiles" ..  || { echo "FAILED CMAKE CONFIGURE" ; exit 1; }
elif [[ "$OSTYPE" == "msys"* ]]; then
    echo "USE MINGW BUILD INSTEAD" ; exit 1
    export INSTALL32="c:/Program Files (x86)"
    export HESAFF_INSTALL="$INSTALL32/Hesaff"
    echo "INSTALL32=$INSTALL32"
    echo "HESAFF_INSTALL=$HESAFF_INSTALL"
    cmake -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX="$HESAFF_INSTALL" -DOpenCV_DIR="$INSTALL32/OpenCV" .. || { echo "FAILED CMAKE CONFIGURE" ; exit 1; }
else
    cmake -G "Unix Makefiles" ..  || { echo "FAILED CMAKE CONFIGURE" ; exit 1; }
fi

if [[ "$OSTYPE" == "msys"* ]]; then
    make || { echo "FAILED MAKE" ; exit 1; }
else
    export NCPUS=$(grep -c ^processor /proc/cpuinfo)
    make -j$NCPUS || { echo "FAILED MAKE" ; exit 1; }
fi

cp libhesaff* ../pyhesaff --verbose
cd ..
