cmake_minimum_required(VERSION 3.13)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.0/bin/nvcc)

enable_language("CUDA")
add_executable(checkThreadIndex checkThreadIndex.cu)
add_definitions(-DUSE_CUDA)

