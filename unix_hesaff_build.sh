cd ~/code/hesaff
mkdir build
cd build

export HOTSPOTTER_DIR=~/code/hotspotter

cmake -G "Unix Makefiles" ..

make
# localize.py should take care of putting hotspotter in the path
export PYTHONPATH=$PYTHONPATH:$HOTSPOTTER_DIR
python $HOTSPOTTER_DIR/hstpl/localize.py
