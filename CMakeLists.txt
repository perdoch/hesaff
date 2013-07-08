cmake_minimum_required(VERSION 2.8)

#load library path
# usr/local/lib ...
# otool -L 
# @executable_path <- probably want todo this

#DYLIB_PATH ENVAR Maybe set? 

project(HessAff)
set(CMAKE_BUILD_TYPE "Release")



message(STATUS "Checking OpenCV")
find_package( OpenCV REQUIRED )

# BEGIN BIG PRINT

message(STATUS "------------------------ ")
message(STATUS "OpenCV Variables    : ")
message(STATUS "find_package(OpenCV Required) ")
message(STATUS "------------------------ ")
message(STATUS "OpenCV_FOUND            : " ${OpenCV_FOUND}) 
message(STATUS "OpenCV_VERSION          : " ${OpenCV_VERSION}) 

# Print Libs 1
message(STATUS "OpenCV_LIBS             1: ")
foreach(cvlib1 ${OpenCV_LIBS}) 
message(STATUS "                        " ${cvlib1})
endforeach(cvlib1)
message(STATUS "------------------------ ")

# Print Include 1
message(STATUS "OpenCV_DIR              1: " ) 
foreach(cvdir1 ${OpenCV_DIR}) 
message(STATUS "                        " ${cvdir1})
endforeach(cvdir1)
message(STATUS "------------------------ ")

# Print Include 2
message(STATUS "mOpenCV_INCLUDE_DIR      2: ") 
foreach(cvdir2 ${OpenCV_INCLUDE_DIR}) 
message(STATUS "                        " ${cvdir2})
endforeach(cvdir2)
message(STATUS "------------------------ ")


message(STATUS "------------------------ ")
message(STATUS "Depricated OpenCV Variables    : ")
message(STATUS "------------------------ ")


# Print Include 3
message(STATUS "OpenCV_INCLUDE_DIRS     3: ") 
foreach(cvdir3 ${OpenCV_INCLUDE_DIRS}) 
message(STATUS "                        " ${cvdir3})
endforeach(cvdir3)
message(STATUS "------------------------ ")

# Print Libs 2
message(STATUS "OpenCV_LIBRARIES        2: ") 
foreach(cvlib2 ${OpenCV_LIBRARIES}) 
message(STATUS "                        " ${cvlib2})
endforeach(cvlib2)
message(STATUS "------------------------ ")

# Print Include 4
message(STATUS "OpenCV_LINK_DIRECTORIES 4: ") 
foreach(cvdir4 ${OpenCV_LINK_DIRECTORIES}) 
message(STATUS "                        " ${cvdir4})
endforeach(cvdir4)
message(STATUS "------------------------ ")


message(STATUS "------------------------ ")
message(STATUS " ")

# END BIG PRINT

message(STATUS "Specifing include directories")

# This should be used but it is not. Weird.
include_directories(${OpenCV_INCLUDE_DIR})
# Apparently this seems to be used
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "Defining hesaff executable")
add_executable(hesaff hesaff.cpp pyramid.cpp helpers.cpp affine.cpp siftdesc.cpp)

message(STATUS "Specifing libraries to link to")
target_link_libraries(hesaff ${OpenCV_LIBS})
