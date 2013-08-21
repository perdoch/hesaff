:: helper variables
set INSTALL32=C:\Program Files (x86)
set CODE=%USERPROFILE%\code
set HESAFF_INSTALL="%INSTALL32%\Hesaff"
set CMAKE_EXE="%INSTALL32%\CMake 2.8\bin\cmake.exe"
set CMAKE_GUI_EXE="%INSTALL32%\CMake 2.8\bin\cmake-gui.exe"
set HESAFF_SRC=%CODE%\hesaff
set HESAFF_BUILD=%CODE%\hesaff\build

echo %HESAFF_INSTALL%
echo %CMAKE_EXE%
echo %HESAFF_BUILD%
echo %HESAFF_SRC%

mkdir %HESAFF_BUILD%
cd %HESAFF_BUILD%

set OPENCV_BIN=%INSTALL32%\OpenCV\bin

set PATH=%OPENCV_BIN%;%PATH%

:: OpenCV settings on windows
%CMAKE_EXE% ^
-G "MSYS Makefiles" ^
-DCMAKE_INSTALL_PREFIX=%HESAFF_INSTALL% ^
-DCMAKE_C_FLAGS=-march=i486 ^
-DCMAKE_CXX_FLAGS=-march=i486 ^
-DOpenCV_DIR="C:\Program Files (x86)\OpenCV" ^
%HESAFF_SRC%

:: make command that doesn't freeze on mingw
mingw32-make -j7 "MAKE=mingw32-make -j3" -f CMakeFiles\Makefile2 all


