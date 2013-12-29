cd ~/code/hesaff
mkdir build
cd build

cmake -G "Unix Makefiles" ..

make

python ~/code/hotspotter/_tpl/localize.py
