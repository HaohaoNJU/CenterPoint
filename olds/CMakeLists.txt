cmake_minimum_required(VERSION 2.8.3)
project(centerpoint)


include_directories(
    ${PROJECT_SOURCE_DIR}/include/common
    ${PROJECT_SOURCE_DIR}/include
)

set(SAMPLES_COMMON_SOURCES
${PROJECT_SOURCE_DIR}/include/common/logger.cpp
)


# pcl and boost related !
find_package(PCL REQUIRED)
# find_package(Boost COMPONENTS program_options REQUIRED )
# include_directories(${Boost_INCLUDE_DIRS})
# link_directories(${Boost_LIBRARY_DIRS})
 
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})



# set flags for CUDA availability
option(CUDA_AVAIL "CUDA available" OFF)
find_package(CUDA)
message("CUDA dir paths", ${CUDA_LIBRARIES})

include_directories( 
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
  )

message("CUDA is available!")
message("CUDA Libs: ${CUDA_LIBRARIES}")
message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
set(CUDA_AVAIL ON)


# set flags for TensorRT availability
option(TRT_AVAIL "TensorRT available" OFF)
# try to find the tensorRT modules
find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVONNXPARSERS NAMES nvonnxparser)

if(NVINFER AND NVPARSERS AND NVONNXPARSERS)
  message("TensorRT is available!")
  message("NVINFER: ${NVINFER}")
  message("NVPARSERS: ${NVPARSERS}")
  message("NVONNXPARSERS: ${NVONNXPARSERS}")
  set(TRT_AVAIL ON)
else()
  message("TensorRT is NOT Available")
  set(TRT_AVAIL OFF)
endif()

# set flags for CUDNN availability
option(CUDNN_AVAIL "CUDNN available" OFF)
# try to find the CUDNN module
find_library(CUDNN_LIBRARY
  NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
  PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR} ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
  PATH_SUFFIXES lib lib64 bin
  DOC "CUDNN library."
)

if(CUDNN_LIBRARY)
  message("CUDNN is available!")
  message("CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
  set(CUDNN_AVAIL ON)
else()
  message("CUDNN is NOT Available")
  set(CUDNN_AVAIL OFF)
endif()


file(GLOB PointPillarLibs ${PROJECT_SOURCE_DIR}/src/*cpp)
add_library(pointpillars SHARED ${PointPillarLibs})

file(GLOB PointPillarCU ${PROJECT_SOURCE_DIR}/src/*cu)
cuda_add_library(pointpillarsCU SHARED ${PointPillarCU})

set(CUDA_LIB_PATH /usr/local/cuda/lib64/)

MESSAGE(${PROJECT_SOURCE_DIR})
add_executable(centerpoint ${PROJECT_SOURCE_DIR}/src/samplecenterpoint.cpp ${SAMPLES_COMMON_SOURCES})
target_link_libraries(centerpoint 
                         ${PCL_LIBRARIES} 
                        #  ${Boost_LIBRARIES} 
                         ${CUDA_LIBRARY} 
                         ${CUDA_RUNTIME_LIBRARY} 
                         nvonnxparser
                         nvinfer
                         pointpillars
                         ${CUDA_LIB_PATH}libcudart.so
                         pointpillarsCU
                         )


