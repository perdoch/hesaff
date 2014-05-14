cmake_minimum_required(VERSION 2.8)

#load library path
# usr/local/lib ...
# otool -L 
# @executable_path <- probably want todo this

#DYLIB_PATH ENVAR Maybe set? 

project(hesaff)
set(CMAKE_BUILD_TYPE "Release")

if (APPLE)
    # Use clang2 on apple for openmp support
    message('Apple System')
    set(CMAKE_C_COMPILER "clang2")
    set(CMAKE_CXX_COMPILER "${CMAKE_C_COMPILER}")

    #MacPorts
    include_directories(/opt/local/include)
    link_directories(/opt/local/lib)

    # Add flags to support clang2
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -stdlib=libstdc++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
    set(CMAKE_SHARED_LINKER_FLAGS "-lstdc++ -stdlib=libstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "-lstdc++ -stdlib=libstdc++")
elseif(WIN32)
    message('Win32 System')
endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
message(STATUS "Checking OpenMP")
find_package(OpenMP REQUIRED)
IF(OPENMP_FOUND)
    message("Found OpenMP! ^_^")
	# add flags for OpenMP
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}	${OpenMP_SHARED_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    #message("CMAKE_C_FLAGS=${CMAKE_C_FLAGS}")
    #message("CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
    #message("CMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}")
    #message("CMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}")
ELSE()
    message("Missed OpenMP! x_x")
ENDIF()


message(STATUS "Checking OpenCV")
find_package( OpenCV REQUIRED )


OPTION(BUILD_HESAFF_SHARED "Build the Foo libraries shared." Off)

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

set(SOURCE_FILES 
    src/hesaff.cpp 
    src/pyramid.cpp 
    src/helpers.cpp 
    src/affine.cpp 
    src/siftdesc.cpp)

add_executable(hesaffexe ${SOURCE_FILES})

message(STATUS "Defining hesaff library")
#add_library(hesaff ${HESAFF_LIBRARY_TYPE} ${SOURCE_FILES})
add_library(hesaff SHARED ${SOURCE_FILES})

message(STATUS "Specifing libraries to link to")
target_link_libraries(hesaff ${OpenCV_LIBS} ${OpenMP_CXX_FLAGS})
target_link_libraries(hesaffexe ${OpenCV_LIBS} ${OpenMP_CXX_FLAGS})
