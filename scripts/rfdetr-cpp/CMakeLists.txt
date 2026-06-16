cmake_minimum_required(VERSION 3.18)
project(rfdetr_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# OpenCV — system install on RHEL (dnf install opencv-devel) or Homebrew on Mac.
find_package(OpenCV 4 REQUIRED)

# ONNX Runtime — no FindOnnxRuntime upstream; resolve include + lib manually.
# Hints cover Homebrew (Mac), system /usr/local, and an ONNXRUNTIME_ROOT override.
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS
        /opt/homebrew/include/onnxruntime
        /opt/homebrew/Cellar/onnxruntime/*/include/onnxruntime
        /usr/local/include/onnxruntime
        /usr/include/onnxruntime
        $ENV{ONNXRUNTIME_ROOT}/include
)
find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS
        /opt/homebrew/lib
        /usr/local/lib
        /usr/lib64
        $ENV{ONNXRUNTIME_ROOT}/lib
)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found. Set ONNXRUNTIME_ROOT or install via brew/dnf.\n"
        "  include: ${ONNXRUNTIME_INCLUDE_DIR}\n"
        "  lib:     ${ONNXRUNTIME_LIB}")
endif()

option(USE_CUDA "Link CUDA Execution Provider (target must have ORT-GPU + CUDA 12.x)" OFF)

add_executable(rfdetr_cpp main.cpp)
target_include_directories(rfdetr_cpp PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(rfdetr_cpp PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})
if(USE_CUDA)
    # The --cuda runtime flag still needs CUDA provider present in libonnxruntime.
    # No extra link target needed beyond the GPU build of libonnxruntime itself.
    target_compile_definitions(rfdetr_cpp PRIVATE RFDETR_USE_CUDA)
endif()
