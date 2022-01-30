#!/bin/bash
# https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html

sudo apt-get install -y build-essential cmake git
#sudo apt-get install python3-dev python3-numpy
sudo apt-get install -y libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev


mkdir -p "$HOME/code/pyhesaff/deps/src"
git clone -b v1.5.0 https://github.com/Kitware/fletch.git "$HOME/code/pyhesaff/deps/src/fletch"

mkdir -p "$HOME/code/pyhesaff/deps/src/fletch/build"
cd "$HOME/code/pyhesaff/deps/src/fletch/build" && 
    cmake \
        -Dfletch_ENABLE_OpenCV=True \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$HOME/code/pyhesaff/deps" \
        -DOpenCV_SELECT_VERSION=4.2.0 .. && \
        make install -j9
