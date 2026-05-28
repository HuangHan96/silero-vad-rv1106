cmake_minimum_required(VERSION 3.16)
project(silero_vad_rv1106 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# RKNN runtime library: defaults to vendored copy in third_party/luckfox-pico,
# override with -DRKNNMRT_LINK_DIR=<path> if needed.
set(RKNNMRT_LINK_DIR
    ${CMAKE_SOURCE_DIR}/third_party/luckfox-pico/media/iva/iva/librockiva/rockiva-rv1106-Linux/lib
    CACHE PATH "Directory containing librknnmrt.so")

add_executable(rknn_vad src/rknn_vad.cpp)
target_include_directories(rknn_vad PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_directories(rknn_vad PRIVATE ${RKNNMRT_LINK_DIR})
target_link_libraries(rknn_vad PRIVATE rknnmrt)
