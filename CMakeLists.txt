project(cudaexec)

cmake_minimum_required(VERSION 3.13)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/bin/nvcc.exe")
enable_language("CUDA")

add_executable(cudaexec checkThreadIndex.cu)
add_definitions(-DUSE_CUDA)

