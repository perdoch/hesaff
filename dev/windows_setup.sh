# https://github.com/cmuratori/meow_hash/blob/master/.travis.yml


# Need to ensure that we install the VC component with vs15
choco install visualstudio2017-workload-vctools -y --package-parameters "--no-includeRecommended --add Microsoft.VisualStudio.Component.VC.x86.amd64 --add Microsoft.VisualStudio.Component.Windows10SDK"
choco install windows-sdk-8.1 -y

# Note: I needed to go through the VS installer GUI and install other various C++ components. Probably could do this via choco, but not sure how

choco install visualstudio2017community -y
choco install visualstudio2017buildtools -y
choco install visualstudio2017-workload-vctools -y

choco install opencv -y

choco install visualstudio2015community -y

#"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe"

#https://stackoverflow.com/questions/43372235/vcvarsall-bat-for-visual-studio-2017
call "C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC/Auxiliary/Build/vcvars64.bat"
cd %HOME%/code/hesaff
#set CMAKE_C_COMPILER=C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC/Tools/MSVC/14.16.27023/bin/HostX86/x86/cl.exe
python setup.py bdist_wheel -- -G "Visual Studio 15 2017 Win64" -DOpenCV_DIR="C:/tools/opencv/build" 

python setup.py bdist_wheel -- -G "Ninja"

python setup.py build_ext --inplace -- -G "Visual Studio 15 2017 Win64" -DOpenCV_DIR="C:/tools/opencv/build"  -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE

set PATH=C:/tools/opencv/build/x64/vc15/bin;%PATH%
#-DBUILD_SHARED_LIBS=TRUE

python setup.py bdist_wheel -- -G "Visual Studio 15 2017 Win64" -DOpenCV_DIR="C:/tools/opencv/build"


choco install opencv -y --forcex86 --force
call "C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC/Auxiliary/Build/vcvars32.bat"
cd %HOME%/code/hesaff
python setup.py build_ext --inplace -- -G "Visual Studio 15 2017" -DOpenCV_DIR="C:/tools/opencv/build"


C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools

"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools\vsdevcmd" -arch=x86


set PYTHON=/c/Users/erote/Miniconda3/
set GENERATOR=Visual Studio 15 2017
echo %PYTHON%
echo %GENERATOR%

set GENERATOR=MinGW Makefiles
python setup.py bdist_wheel -- -G "%GENERATOR%" -DOpenCV_DIR="C:/tools/opencv/build"
python setup.py bdist_wheel -- -G "MinGW Makefiles" -DOpenCV_DIR="C:/tools/opencv/build"
python setup.py bdist_wheel -- -G "Ninja" -DOpenCV_DIR="C:/tools/opencv/build"
