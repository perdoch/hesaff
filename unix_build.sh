#cd ~/code/hesaff
mkdir build
cd build

cmake -G "Unix Makefiles" .. && make

cp libhesaff* ../pyhesaff
