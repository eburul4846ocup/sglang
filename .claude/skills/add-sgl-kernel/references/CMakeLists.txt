cmake_minimum_required(VERSION 3.18)
project(scale_kernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find required packages
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)

# Include directories
include_directories(
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Collect source files
set(SOURCES
    scale_kernel.cu
    scale_kernel_binding.cpp
)

# Build shared library
add_library(scale_kernel SHARED ${SOURCES})

# Link against PyTorch and Python
target_link_libraries(scale_kernel
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
)

# Set output name to match Python import expectations
set_target_properties(scale_kernel PROPERTIES
    PREFIX ""
    OUTPUT_NAME "scale_kernel"
    SUFFIX ".so"
)

# Enable position-independent code
set_target_properties(scale_kernel PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# CUDA architecture flags — adjust for your target GPU
set_property(TARGET scale_kernel PROPERTY CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Compiler flags
target_compile_options(scale_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        --expt-relaxed-constexpr
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -Wall
    >
)
