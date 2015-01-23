#cd ~/code/hesaff
echo "checking if build dir should be removed"

#RMBUILD=1
#for i in "$@"
#do
#case $i in --fast)
    #RMBUILD=0
    #;;
#esac
#done
#if [[ "$RMBUILD" == "1" ]]; then
    #rm -rf build
#fi
#python -c "import utool as ut; print('keeping build dir' if ut.get_argflag('--fast') else ut.delete('build'))" $@

# +==================================================
# SIMPLE WAY OF EXECUTING MULTILINE PYTHON FROM BASH
# +--------------------------------------------------
# Creates custom file descriptor that runs the script
# References: http://superuser.com/questions/607367/raw-multiline-string-in-bash
exec 42<<'__PYSCRIPT__'
import utool as ut;

if not ut.get_argflag('--fast'):
    print('deleting build dir')
    ut.delete('build')
else:
    print('keeping build dir')
__PYSCRIPT__
python /dev/fd/42 $@
# L_________________________________________________

mkdir build
cd build

if [[ "$OSTYPE" == "darwin"* ]]; then
    cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 -G "Unix Makefiles" ..  || { echo "FAILED CMAKE CONFIGURE" ; exit 1; }
else
    cmake -G "Unix Makefiles" ..  || { echo "FAILED CMAKE CONFIGURE" ; exit 1; }
fi

export NCPUS=$(grep -c ^processor /proc/cpuinfo)
make -j$NCPUS || { echo "FAILED MAKE" ; exit 1; }

cp libhesaff* ../pyhesaff --verbose
cd ..
