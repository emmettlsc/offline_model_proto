cmake_minimum_required(VERSION 3.18)
project(sam2_trt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV 4 REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS
        $ENV{TENSORRT_ROOT}/include
        /usr/local/TensorRT/include
        /opt/TensorRT/include
        /usr/include
)
find_library(TENSORRT_NVINFER nvinfer
    HINTS
        $ENV{TENSORRT_ROOT}/lib
        $ENV{TENSORRT_ROOT}/lib64
        /usr/local/TensorRT/lib
        /opt/TensorRT/lib
        /usr/lib64
)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER)
    message(FATAL_ERROR
        "tensorrt not found. set TENSORRT_ROOT.\n"
        "  include: ${TENSORRT_INCLUDE_DIR}\n"
        "  lib:     ${TENSORRT_NVINFER}")
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(sam2_trt main.cpp)
target_include_directories(sam2_trt PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(sam2_trt PRIVATE
    ${OpenCV_LIBS}
    ${TENSORRT_NVINFER}
    CUDA::cudart)
