# We need to build opencv explicitly on 32bit systems
pip install cmake ninja

choco install 7zip -y
choco install wget -y

wget https://github.com/opencv/opencv/archive/4.1.0.zip 
# extracts to opencv-4.1.0
7z x 4.1.0.zip

#set ARCH=Win32
#call "C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC/Auxiliary/Build/vcvars32.bat"

set ARCH=Win64
call "C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC/Auxiliary/Build/vcvars64.bat"

cd %HOME%/code/hesaff

mkdir "opencv-4.1.0\build_%ARCH%"
cd "opencv-4.1.0\build_%ARCH%"

#cmake -G "Ninja" -DBUILD_opencv_apps=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DINSTALL_CREATE_DISTRIB=ON -DWITH_WIN32UI=OFF -DWITH_QT=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=TRUE ..

cmake -G "Ninja" -DBUILD_opencv_apps=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_DOCS=OFF -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DINSTALL_CREATE_DISTRIB=ON -DWITH_WIN32UI=OFF -DWITH_QT=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=TRUE -DBUILD_PROTOBUF=Off  -DBUILD_opencv_imgcodecs=On  -DBUILD_opencv_imgproc=On  -DBUILD_opencv_calib3d=Off  -DBUILD_opencv_dnn=Off  -DBUILD_opencv_features2d=Off  -DBUILD_opencv_flann=Off  -DBUILD_opencv_gapi=Off  -DBUILD_opencv_ml=Off  -DBUILD_opencv_highgui=On  -DBUILD_opencv_java_bindings_generator=Off  -DBUILD_opencv_python_bindings_generator=Off  -DBUILD_opencv_python_tests=Off  -DBUILD_opencv_objdetect=Off  -DBUILD_opencv_photo=Off  -DBUILD_opencv_stitching=Off  -DBUILD_opencv_video=Off  -DBUILD_opencv_videoio=Off  -DBUILD_opencv_world=Off  ..



ninja
# on windows seems to install to the <CMAKE_BUILD_DIR>/install
ninja install

#-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=TRUE
#cd opencv-4.1.0/build

cd ../..
#python setup.py build_ext --inplace -- -G "Ninja" -DOpenCV_DIR="opencv-4.1.0/build_%ARCH%/install" -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=TRUE
python setup.py build_ext --inplace -- -G "Visual Studio 15 2017 %ARCH%" -DOpenCV_DIR="opencv-4.1.0/build_%ARCH%/install" -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DBUILD_SHARED_LIBS=TRUE


dumpbin /dependents pyhesaff/libhesaff.win-amd64-3.6.dll


set PATH=C:\tools\opencv\build\x64\vc15\bin;%PATH%
python -c "import ctypes; print(ctypes.cdll['pyhesaff/libhesaff.win-amd64-3.6.dll'])"

python -c "import ctypes, os; print(ctypes.windll[os.path.abspath(os.path.normpath('_skbuild/win-amd64-3.6/cmake-build/Release/hesaff.dll'))])"
python -c "import ctypes, os; print(os.path.abspath(os.path.normpath('_skbuild/win-amd64-3.6/cmake-build/Release/hesaff.dll')))"
python -c "import ctypes; print(ctypes.cdll['_skbuild/win-amd64-3.6/cmake-build/Release/hesaff.dll'])"
