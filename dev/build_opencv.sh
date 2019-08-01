# STARTBLOCK bash

 # https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html

sudo apt-get install build-essential cmake git
#sudo apt-get install python3-dev python3-numpy
sudo apt-get install libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev


cd $HOME/code
git clone https://github.com/Itseez/opencv.git
cd $HOME/code/opencv
git pull

#if [[ "$VIRTUAL_ENV" == ""  ]]; then
#    # The case where we are installying system-wide
#    # It is recommended that a virtual enviornment is used instead
#    export PYTHON_EXECUTABLE=$(which {pyversion})
#    if [[ '$OSTYPE' == 'darwin'* ]]; then
#        # Mac system info
#        export LOCAL_PREFIX=/opt/local
#        export PYTHON3_PACKAGES_PATH=$($PYTHON_EXECUTABLE -c "import site; print(site.getsitepackages()[0])")
#        export _SUDO="sudo"
#    else
#        # Linux system info
#        export LOCAL_PREFIX=/usr/local
#        export PYTHON_PACKAGES_PATH=$LOCAL_PREFIX/lib/{pyversion}/dist-packages
#        export _SUDO="sudo"
#    fi
#    # No windows support here
#else
#    # The prefered case where we are in a virtual environment
#    export PYTHON_EXECUTABLE=$(which python)
#    # export LOCAL_PREFIX=$VIRTUAL_ENV/local
#    export LOCAL_PREFIX=$VIRTUAL_ENV
#    pyversion=3.6
#    export PYTHON_PACKAGES_PATH=$LOCAL_PREFIX/lib/{pyversion}/site-packages
#    export _SUDO=""
#fi

#mkdir -p $LOCAL_PREFIX


LOCAL_PREFIX=$HOME/.local
echo "LOCAL_PREFIX = $LOCAL_PREFIX"
        

# Checkout opencv core
# git clone https://github.com/Itseez/opencv.git
# Checkout opencv extras
#git clone https://github.com/Itseez/opencv_contrib.git
# cd opencv_contrib
# git pull
# cd ..
# git pull
build_dname=build
REPO_DIR=$HOME/code/opencv
echo "REPO_DIR = $REPO_DIR"
mkdir -p $REPO_DIR/$build_dname
cd $REPO_DIR/$build_dname


cd $HOME/code/opencv/build
cmake -G "Ninja" \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=$HOME/.local \
    -D BUILD_ZLIB=On \
    -D BUILD_PNG=On \
    ..
ninja

#cmake -G "Ninja" \
#    -D WITH_OPENMP=ON \
#    -D CMAKE_BUILD_TYPE=RELEASE \
#    -D BUILD_opencv_python3=On \
#    -D BUILD_opencv_python2=Off \
#    -D BUILD_OPENCV_PYTHON3_VERSION=On \
#    -D CMAKE_INSTALL_PREFIX=$LOCAL_PREFIX \
#    -D WITH_CUDA=Off \
#    -D WITH_FFMPEG=Off \
#    -D WITH_GSTREAMER=Off \
#    -D WITH_GTK=Off \
#    -D WITH_VTK=Off \
#    -D WITH_1394=Off \
#    -D WITH_CUDA=Off \
#    -D BUILD_opencv_java_bindings_generator=Off \
#    -D BUILD_opencv_objectdetect=Off \
#    -D BUILD_opencv_python_tests=Off \
#    -D BUILD_ZLIB=On \
#    -D BUILD_PNG=On \
#    -D BUILD_TIFF=On \
#    -D WITH_MATLAB=Off \
#    -D BUILD_opencv_dnn=Off \
#    $REPO_DIR
#    #-D OPENCV_EXTRA_MODULES_PATH=$REPO_DIR/opencv_contrib/modules \
#    #-D BUILD_opencv_dnn_modern=Off \
#    #-D PYTHON_PACKAGES_PATH=${PYTHON_PACKAGES_PATH} \
#    # -D WITH_OPENCL=Off \
#    # -D BUILD_opencv_face=Off \
#    # -D BUILD_opencv_objdetect=Off \
#    # -D BUILD_opencv_video=Off \
#    # -D BUILD_opencv_videoio=Off \
#    # -D BUILD_opencv_videostab=Off \
#    # -D BUILD_opencv_ximgproc=Off \
#    # -D BUILD_opencv_xobjdetect=Off \
#    # -D BUILD_opencv_xphoto=Off \
#    # -D BUILD_opencv_datasets=Off \
#    # -D CXX_FLAGS="-std=c++11" \ %TODO
#export NCPUS=$(grep -c ^processor /proc/cpuinfo)
#ninja

#ninja -j$NCPUS
