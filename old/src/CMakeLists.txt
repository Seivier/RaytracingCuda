

add_library(Raytracing SHARED
        Color.cu
        Ray.cu
        Hittable.cu
        Sphere.cu
        HittableList.cu
        Camera.cu
        Vector.cu
        Material.cu
        Renderer.cu)

set_target_properties(Raytracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(Raytracing PROPERTIES POSITION_INDEPENDENT_CODE ON)
#target_link_libraries(Raytracing glfw glad imgui glm)

# link cuda random library
target_include_directories(Raytracing PUBLIC ${INCLUDE_DIR})

#install(TARGETS Raytracing DESTINATION lib)