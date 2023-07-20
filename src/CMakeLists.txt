

add_library(Raytracing STATIC
        Color.cu
        Ray.cu
        Utility.cu
        Hittable.cu
        Sphere.cu
        HittableList.cu
        Camera.cu
        Vector.cu
        Material.cu ../include/Raytracing/Material.cuh Color.cu ../include/Raytracing/Color.cuh)

#set_target_properties(Raytracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(Raytracing glfw glad imgui glm)
target_include_directories(Raytracing PUBLIC ${INCLUDE_DIR})

#install(TARGETS Raytracing DESTINATION lib)