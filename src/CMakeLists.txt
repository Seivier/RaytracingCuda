

set(RT_SOURCES "rt/vector.cu" "rt/ray.cu" "rt/hittable.cu" "rt/sphere.cu" "rt/hittable_list.cu" "rt/camera.cu" "rt/material.cu")
set(RT_HEADERS "rt/vector.cuh" "rt/ray.cuh" "rt/hittable.cuh" "rt/sphere.cuh" "rt/hittable_list.cuh" "rt/camera.cuh" "rt/material.cuh")
add_executable(rtcuda main.cu ${RT_SOURCES} ${RT_HEADERS})
set_target_properties(rtcuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)