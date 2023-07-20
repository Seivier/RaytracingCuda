cmake_minimum_required(VERSION 3.8)
project(RayTracingCUDA LANGUAGES CUDA CXX)


set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(RT_HEADERS
        "${INCLUDE_DIR}/Raytracing/Camera.cuh"
        "${INCLUDE_DIR}/Raytracing/Color.cuh"
        "${INCLUDE_DIR}/Raytracing/Hittable.cuh"
        "${INCLUDE_DIR}/Raytracing/HittableList.cuh"
        "${INCLUDE_DIR}/Raytracing/Material.cuh"
        "${INCLUDE_DIR}/Raytracing/Ray.cuh"
        "${INCLUDE_DIR}/Raytracing/Sphere.cuh"
        "${INCLUDE_DIR}/Raytracing/Vector.cuh"
        "${INCLUDE_DIR}/Raytracing/Utils.cuh"
        )


add_subdirectory(extern)
add_subdirectory(src)
add_subdirectory(demo)

#add_executable(main main.cpp)
#
#set_target_properties(main PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)
