#cd ~/code/hesaff
echo "removing old build"
rm -rf build

mkdir build
cd build

if [[ "$OSTYPE" == "darwin"* ]]; then
    cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 -G "Unix Makefiles" ..  || { echo "FAILED CMAKE CONFIGURE" ; exit 1; }
else
    cmake -G "Unix Makefiles" ..  || { echo "FAILED CMAKE CONFIGURE" ; exit 1; }
fi

export NCPUS=$(grep -c ^processor /proc/cpuinfo)
make -j$NCPUS || { echo "FAILED MAKE" ; exit 1; }

cp build/libhesaff* ../pyhesaff --verbose
cd ..
