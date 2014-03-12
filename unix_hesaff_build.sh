cd ~/code/hesaff
mkdir build
cd build

export HOTSPOTTER_DIR=~/code/hotspotter

# need better way of setting hotspotterdir or not setting it at all
export PYTHONPATH=$PYTHONPATH:$HOTSPOTTER_DIR

cmake -G "Unix Makefiles" .. && make && python $HOTSPOTTER_DIR/hstpl/localize.py
