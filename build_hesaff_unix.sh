cd ~/code/hesaff
mkdir build
cd build

cmake -G "Unix Makefiles" ..

make
# localize.py should take care of putting hotspotter in the path
#export PYTHONPATH=$PYTHONPATH:~/code/  
python ~/code/hotspotter/hstpl/localize.py
