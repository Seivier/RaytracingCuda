cmake_minimum_required(VERSION 3.8)
project(RayTracingCUDA LANGUAGES CXX CUDA C)


set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_subdirectory(extern)
add_subdirectory(src)


