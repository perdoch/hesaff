cmake_minimum_required(VERSION 2.8)

# Clang2 = Clang + OpenMP built for OSX Mavericks
# http://goo.gl/1Tg0Zj

#######################################
if (APPLE AND CLANG2)
    SET(CLANG2 Off)
	message('Apple System')
    set(CMAKE_C_COMPILER "clang2")
    set(CMAKE_CXX_COMPILER "${CMAKE_C_COMPILER}")
else()
    SET(CLANG2 On)
endif()

if (APPLE)
    #MacPorts
    include_directories(/opt/local/include)
    link_directories(/opt/local/lib)
endif()

#######################################
project(hesaff)
set(CMAKE_BUILD_TYPE "Release")

# generate debug info for gdb
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")

#######################################
message(STATUS "Checking OpenCV")
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

#######################################
message(STATUS "Checking OpenMP")
find_package(OpenMP REQUIRED)
IF(OPENMP_FOUND)
    message("Found OpenMP! ^_^")
    # add flags for OpenMP
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_C_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ELSE()
    message("Missed OpenMP! x_x")
ENDIF()

#######################################
if(APPLE AND CLANG2)    
    # Add flags to support clang2
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -stdlib=libstdc++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lstdc++ -stdlib=libstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++ -stdlib=libstdc++")
endif()

#######################################
set(SOURCE_FILES 
    src/hesaff.cpp 
    src/pyramid.cpp 
    src/helpers.cpp 
    src/affine.cpp 
    src/siftdesc.cpp)

add_executable(hesaffexe ${SOURCE_FILES})
add_library(hesaff SHARED ${SOURCE_FILES})

#######################################
target_link_libraries(hesaff ${OpenCV_LIBS})
target_link_libraries(hesaffexe ${OpenCV_LIBS})
