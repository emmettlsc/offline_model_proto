cmake_minimum_required(VERSION 3.18)
project(sam2_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV 4 REQUIRED)

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

foreach(tgt sam2_cpp:main.cpp sam2_stream:stream.cpp)
    string(REPLACE ":" ";" parts ${tgt})
    list(GET parts 0 name)
    list(GET parts 1 src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(${name} PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})
    if(USE_CUDA)
        target_compile_definitions(${name} PRIVATE SAM2_USE_CUDA)
    endif()
endforeach()
