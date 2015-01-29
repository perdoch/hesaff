cmake_minimum_required(VERSION 2.8)

#######################################
# Clang2 = Clang + OpenMP built for OSX Mavericks
# http://goo.gl/1Tg0Zj
if (APPLE)
    message("Detected APPLE system")
    SET(CLANG2 Off)
endif()

if (APPLE AND CLANG2)
    message("Using clang2")
    set(CMAKE_C_COMPILER "clang2")
    set(CMAKE_CXX_COMPILER "clang2++")
endif()

if (FALSE)
    set(CMAKE_C_COMPILER "clang-3.5")
    set(CMAKE_CXX_COMPILER "clang++-3.5")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -stdlib=libstdc++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++ -lstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++ -lstdc++ ")
endif()

#######################################
project(hesaff) #### !!!!IMPORTANT!!!! THIS MUST BE DOWN HERE FOR CLANG2
set(CMAKE_BUILD_TYPE "Release")

if (APPLE)
    #MacPorts
    message("USING MACPORTS")
    include_directories(/opt/local/include)
    link_directories(/opt/local/lib)
endif()

#######################################
message(STATUS "Checking OpenCV")
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

#######################################
if(APPLE AND CLANG2)    
    # Add flags to support clang2
    message("APPLE + CLANG2: Adding stdlib flags for clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -stdlib=libstdc++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++ -lstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++ -lstdc++ ")
endif()

# # generate debug info for gdb
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")

#######################################
message(STATUS "Checking OpenMP")
find_package(OpenMP)
IF(OPENMP_FOUND)
    message("Found OpenMP! ^_^")
    # add flags for OpenMP
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_C_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ELSE()
    message("Missed OpenMP! x_x")
ENDIF()

#######################################
set(SOURCE_FILES 
    src/hesaff.cpp 
    src/pyramid.cpp 
    src/helpers.cpp 
    src/affine.cpp 
    src/orientation.cpp 
    #src/python_bindings.cpp 
    src/siftdesc.cpp)

add_library(hesaff SHARED ${SOURCE_FILES})
add_executable(hesaffexe ${SOURCE_FILES})

#######################################
target_link_libraries(hesaff ${OpenCV_LIBS})
target_link_libraries(hesaffexe ${OpenCV_LIBS})
