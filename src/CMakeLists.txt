cmake_minimum_required(VERSION 2.8.3)
project(centerpoint)

set(USE_CUDA True)

# For TensorRT sample lib
# set(TRT_ROOT  /home/wanghao/Desktop/projects/TensorRT)

include_directories(
 common
)
set(SAMPLES_COMMON_SOURCES
    common/logger.cpp
)


#set(SAMPLE_SOURCES samplecenterpoint.cpp)
#set(SAMPLE_PARSERS "onnx")
#include(
#~/TensorRT/samples/CMakeSamplesTemplate.txt
# )


# pcl and boost related !
find_package(PCL REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED )
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
 
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

if (USE_CUDA)
  message("CUDA is available!")
  message("CUDA Libs: ${CUDA_LIBRARIES}")
  message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
  set(CUDA_AVAIL ON)
else()
  message("CUDA NOT FOUND OR INCOMPATIBLE CMAKE VERSION FOUND")
  set(CUDA_AVAIL OFF)
endif ()

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

# Check TVM utility, runtime libraries and TVM files
find_library(TVM_LIBRARY NAMES tvm_runtime)
set(TVM_AVAIL OFF)
set(TVM_PFE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/tvm_models/tvm_point_pillars_pfe/)
set(TVM_RPN_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/tvm_models/tvm_point_pillars_rpn/)

if (EXISTS "${TVM_PFE_FOLDER}/deploy_graph.json"
    AND EXISTS "${TVM_PFE_FOLDER}/deploy_param.paranmsNormalLauncherms"
    AND EXISTS "${TVM_PFE_FOLDER}/deploy_lib.so"
    AND EXISTS "${TVM_PFE_FOLDER}/inference_engine_tvm_config.hpp"
    AND EXISTS "${TVM_RPN_FOLDER}/deploy_graph.json"
    AND EXISTS "${TVM_RPN_FOLDER}/deploy_param.params"
    AND EXISTS "${TVM_RPN_FOLDER}/deploy_lib.so"
    AND EXISTS "${TVM_RPN_FOLDER}/inference_engine_tvm_config.hpp")
  set(TVM_AVAIL true)
endif()

file(GLOB PointPillarLibs ${PROJECT_SOURCE_DIR}/**cpp)
add_library(pointpillars SHARED ${PointPillarLibs})

file(GLOB PointPillarCU ${PROJECT_SOURCE_DIR}/**cu)
cuda_add_library(pointpillarsCU SHARED ${PointPillarCU})

set(CUDA_LIB_PATH /usr/local/cuda/lib64/)

add_executable(centerpoint samplecenterpoint.cpp ${SAMPLES_COMMON_SOURCES})
target_link_libraries(centerpoint 
                         ${PCL_LIBRARIES} 
                         ${Boost_LIBRARIES} 
                         ${CUDA_LIBRARY} 
                         ${CUDA_RUNTIME_LIBRARY} 
                         nvonnxparser
                         nvinfer
                         pointpillars
                         ${CUDA_LIB_PATH}libcudart.so
                         pointpillarsCU
                         )


#add_executable(samplerpn samplerpn.cpp ${SAMPLES_COMMON_SOURCES})
#target_link_libraries(samplerpn 
#                         ${PCL_LIBRARIES} 
#                         ${Boost_LIBRARIES} 
#                         ${CUDA_LIBRARY} 
#                         ${CUDA_RUNTIME_LIBRARY} 
#                         nvonnxparser
#                         nvinfer
#                         pointpillars
#                         ${CUDA_LIB_PATH}libcudart.so
#                         #pointpillarsCU
#                         )




#file(GLOB PointPillarLibs  ${PROJECT_SOURCE_DIR}/nodes/**)
##file(GLOB PointPillarCU  ${PROJECT_SOURCE_DIR}/nodes/**cu)
#
#message("PointPillarLibsNodes",${PointPillarLibs})
##add_library(point_pillars SHARED ${PointPillarLibs})
#cuda_add_library(point_pillars SHARED ${PointPillarLibs})
#
##link_directories(${PROJECT_SOURCE_DIR}/nodes)
#link_libraries ( 
#                         point_pillars
#                        ${PCL_LIBRARIES} 
#                        ${Boost_LIBRARIES} 
#                        ${CUDA_LIBRARY} 
#                        ${CUDA_RUNTIME_LIBRARY} 
#                        )
#add_executable(lidar_point_pillars ${PROJECT_SOURCE_DIR}/nodes/lidar_point_pillar_test.cpp)
#
#target_link_libraries (lidar_point_pillars 
#                        ${PCL_LIBRARIES} 
#                        ${Boost_LIBRARIES} 
#                        ${CUDA_LIBRARY} 
#                        ${CUDA_RUNTIME_LIBRARY} 
#                        point_pillars
#                        nvonnxparser
#                        )
 


