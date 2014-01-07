cd ~/code/hesaff
mkdir build
cd build

cmake -G "Unix Makefiles" ..

make
export PYTHONPATH=$PYTHONPATH:~/code/
python ~/code/hotspotter/_tpl/localize.py
