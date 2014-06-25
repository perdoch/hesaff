#cd ~/code/hesaff
mkdir build
cd build

if [[ "$OSTYPE" == "darwin"* ]]; then
    cmake -DCMAKE_OSX_ARCHITECTURES=x86_64 -G "Unix Makefiles" .. && make
else
    cmake -G "Unix Makefiles" .. && make

cp libhesaff* ../pyhesaff
