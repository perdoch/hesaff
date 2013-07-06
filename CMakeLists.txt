cmake_minimum_required(VERSION 2.8)

#load library path
# usr/local/lib ...
# otool -L 
# @executable_path <- probably want todo this

#DYLIB_PATH ENVAR Maybe set? 

project(HessAff)

find_package( OpenCV REQUIRED )

add_executable(hesaff hesaff.cpp pyramid.cpp helpers.cpp affine.cpp siftdesc.cpp)
target_link_libraries(hesaff ${OpenCV_LIBS})
