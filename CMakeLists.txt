cmake_minimum_required(VERSION 2.8)

#load library path
# usr/local/lib ...
# otool -L 
# @executable_path <- probably want todo this

#DYLIB_PATH ENVAR Maybe set? 

project(hesaff)
set(CMAKE_BUILD_TYPE "Release")



message(STATUS "Checking OpenCV")
find_package( OpenCV REQUIRED )


OPTION(BUILD_HESAFF_SHARED "Build the Foo libraries shared." On)

IF(BUILD_HESAFF_SHARED)
  SET(HESAFF_LIBRARY_TYPE SHARED)
ELSE(BUILD_HESAFF_SHARED)
    SET(HESAFF_LIBRARY_TYPE STATIC)
ENDIF(BUILD_HESAFF_SHARED)

# This should be used but it is not. Weird.
include_directories(${OpenCV_INCLUDE_DIR})
# Apparently this seems to be used
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "Defining ell_desc executable")
add_executable(hesaffexe hesaff.cpp pyramid.cpp helpers.cpp affine.cpp siftdesc.cpp)

message(STATUS "Defining hesaff library")
#add_library(hesaff ${HESAFF_LIBRARY_TYPE} hesaff.cpp pyramid.cpp helpers.cpp affine.cpp siftdesc.cpp)
add_library(hesaff SHARED hesaff.cpp pyramid.cpp helpers.cpp affine.cpp siftdesc.cpp)


message(STATUS "Specifing libraries to link to")
target_link_libraries(hesaff ${OpenCV_LIBS})
target_link_libraries(hesaffexe ${OpenCV_LIBS})

