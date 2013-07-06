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
message(STATUS "------------------------ ")
message(STATUS "find_package(OpenCV Required) ")
message(STATUS "------------------------ ")
message(STATUS "OpenCV_FOUND            : " ${OpenCV_FOUND}) 
message(STATUS "OpenCV_VERSION          : " ${OpenCV_VERSION}) 
message(STATUS "OpenCV_LIBS             : " ${OpenCV_LIBS}) 
message(STATUS "OpenCV_DIR              : " ${OpenCV_DIR}) 
message(STATUS "mOpenCV_INCLUDE_DIR      : " ${OpenCV_INCLUDE_DIR}) 
message(STATUS "------------------------ ")
message(STATUS "Depricated Variables    : ")

message(STATUS "OpenCV_INCLUDE_DIRS     : " ${OpenCV_INCLUDE_DIRS}) 
message(STATUS "OpenCV_LIBRARIES        : " ${OpenCV_LIBRARIES}) 
message(STATUS "OpenCV_LINK_DIRECTORIES : " ${OpenCV_LINK_DIRECTORIES}) 
message(STATUS "------------------------ ")
message(STATUS " ")

# Because I'm fed up with it
set(MY_OpenCV_INCLUDE_DIR 
    C:/Users/jon.crall/code/opencv/build
    C:/Users/jon.crall/code/opencv/include
    C:/Users/jon.crall/code/opencv/include/opencv
    C:/Users/jon.crall/code/opencv/modules/core/include
    C:/Users/jon.crall/code/opencv/modules/imgproc/include
    C:/Users/jon.crall/code/opencv/modules/flann/include
    C:/Users/jon.crall/code/opencv/modules/highgui/include
    C:/Users/jon.crall/code/opencv/modules/features2d/include
    C:/Users/jon.crall/code/opencv/modules/calib3d/include
    C:/Users/jon.crall/code/opencv/modules/ml/include
    C:/Users/jon.crall/code/opencv/modules/video/include
    C:/Users/jon.crall/code/opencv/modules/objdetect/include
    C:/Users/jon.crall/code/opencv/modules/contrib/include
    C:/Users/jon.crall/code/opencv/modules/gpuarithm/include
    C:/Users/jon.crall/code/opencv/modules/gpuwarping/include
    C:/Users/jon.crall/code/opencv/modules/gpu/include
    C:/Users/jon.crall/code/opencv/modules/legacy/include
    C:/Users/jon.crall/code/opencv/modules/gpufilters/include
    C:/Users/jon.crall/code/opencv/modules/gpuimgproc/include
    C:/Users/jon.crall/code/opencv/modules/gpubgsegm/include
    C:/Users/jon.crall/code/opencv/modules/gpucodec/include
    C:/Users/jon.crall/code/opencv/modules/gpufeatures2d/include
    C:/Users/jon.crall/code/opencv/modules/gpuoptflow/include
    C:/Users/jon.crall/code/opencv/modules/gpustereo/include
    C:/Users/jon.crall/code/opencv/modules/nonfree/include
    C:/Users/jon.crall/code/opencv/modules/photo/include
    C:/Users/jon.crall/code/opencv/modules/softcascade/include
    C:/Users/jon.crall/code/opencv/modules/stitching/include
    C:/Users/jon.crall/code/opencv/modules/superres/include
    C:/Users/jon.crall/code/opencv/modules/ts/include
    C:/Users/jon.crall/code/opencv/modules/videostab/include)
set(MY_OpenCV_LIBS
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_calib3d249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_contrib249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_core249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_features2d249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_flann249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpu249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpuarithm249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpubgsegm249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpucodec249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpufeatures2d249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpufilters249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpuimgproc249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpuoptflow249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpustereo249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpuwarping249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_highgui249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_imgproc249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_legacy249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_ml249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_nonfree249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_objdetect249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_photo249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_softcascade249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_stitching249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_superres249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_ts249d.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_video249d.dll.a
    debugC:/Users/jon.crall/code/opencv/build/lib/libopencv_videostab249d.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_calib3d249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_contrib249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_core249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_features2d249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_flann249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpu249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpuarithm249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpubgsegm249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpucodec249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpufeatures2d249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpufilters249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpuimgproc249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpuoptflow249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpustereo249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_gpuwarping249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_highgui249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_imgproc249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_legacy249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_ml249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_nonfree249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_objdetect249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_photo249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_softcascade249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_stitching249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_superres249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_ts249.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_video249.dll.a
    optimizedC:/Users/jon.crall/code/opencv/build/lib/libopencv_videostab249.dll.a)


message(STATUS "Specifing include directories")

# This should be used but it is not. Weird.
#include_directories(${OpenCV_INCLUDE_DIR})
# Apparently this seems to be used
#include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${MY_OpenCV_INCLUDE_DIR})

message(STATUS "Defining hesaff executable")
add_executable(hesaff hesaff.cpp pyramid.cpp helpers.cpp affine.cpp siftdesc.cpp)

message(STATUS "Specifing libraries to link to")
#target_link_libraries(hesaff ${OpenCV_LIBS})
target_link_libraries(hesaff ${MY_OpenCV_LIBS})
