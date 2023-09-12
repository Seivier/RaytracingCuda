#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include <cuda_runtime_api.h>

#include "rt/vector.cuh"
#include "rt/ray.cuh"
#include "rt/sphere.cuh"
#include "rt/hittable_list.cuh"
#include "rt/camera.cuh"
#include "rt/material.cuh"


#define CUDA_CALL(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#define GL_CALL(val) do { val; check_gl(#val, __FILE__, __LINE__); } while(0)


void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

void check_glfw(int error, const char* description)
{
	std::cerr << "GLFW error = " << error << " : " << description << "\n";
}

void check_gl(char const *const func, const char *const file, int const line) {
	GLenum error = glGetError();
	if (error != GL_NO_ERROR) {
		std::cerr << "GL error = " << error << " at " <<
			file << ":" << line << " '" << func << "' \n";
		exit(99);
	}
}

__device__ vector rayColor(const ray& r, hittable **world, int depth, curandState *localRand) {
    ray curRay = r;
    vector curAttenuation = vector(1.0,1.0,1.0);
    for(int i = 0; i < depth; i++) {
        hit_record rec;
        if ((*world)->hit(curRay, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vector attenuation;
            if(rec.matPtr->scatter(curRay, rec, attenuation, scattered, localRand)) {
                curAttenuation *= attenuation;
                curRay = scattered;
            }
            else {
                return vector(0.0,0.0,0.0);
            }
        }
        else {
            vector unitDirection = unitVector(curRay.direction());
            float t = 0.5f*(unitDirection.y() + 1.0f);
            vector c = (1.0f-t)*vector(1.0, 1.0, 1.0) + t*vector(0.5, 0.7, 1.0);
            return curAttenuation * c;
        }
    }
    return vector(0.0,0.0,0.0); // exceeded recursion
}

__global__ void randInit(curandState *randState) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, randState);
    }
}

__global__ void renderInit(int px, int py, curandState *randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= px) || (j >= py)) return;
    int pixel_index = j*px + i;
    curand_init(1984+pixel_index, 0, 0, &randState[pixel_index]);
}

__global__ void render(float *pixels, int px, int py, int ns, int depth, camera **cam, hittable **world, curandState *randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= px) || (j >= py)) return;
    int pIdx = j*px + i;
    curandState localRand = randState[pIdx];
	float u = float(i + curand_uniform(&localRand)) / float(px);
	float v = float(j + curand_uniform(&localRand)) / float(py);
	ray r = (*cam)->getRay(u, v, &localRand);
	color col = rayColor(r, world, depth, &localRand);
    randState[pIdx] = localRand;
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
	pixels[3*pIdx] = col.r()*1.0f/ns + pixels[3*pIdx]* (1.0f - 1.0f/ns);
	pixels[3*pIdx+1] = col.g()*1.0f/ns + pixels[3*pIdx+1]* (1.0f - 1.0f/ns);
	pixels[3*pIdx+2] = col.b()*1.0f/ns + pixels[3*pIdx+2]* (1.0f - 1.0f/ns);
//	pixels[3*pIdx] = col.r();
//	pixels[3*pIdx+1] = col.g();
//	pixels[3*pIdx+2] = col.b();
}

#define RND (curand_uniform(&localRand))

__global__ void createWorld(hittable **world, camera **cam, int px, int py, curandState *randState) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState localRand = *randState;
        auto w = new hittable_list();
		w->add(new sphere(vector(0,-1000,0), 1000, new lambertian(vector(0.5, 0.5, 0.5))));
		for (int a = -11; a < 11; a++)
		{
			for (int b = -11; b < 11; b++)
			{
				float choose_mat = RND;
				vector center(a + 0.9*RND, 0.2, b + 0.9*RND);
				if ((center - vector(4, 0.2, 0)).length() > 0.9)
				{
					if (choose_mat < 0.8) // diffuse
					{
						w->add(new sphere(center, 0.2, new lambertian(vector(RND*RND, RND*RND, RND*RND))));
					}
					else if (choose_mat < 0.95) // metal
					{
						w->add(new sphere(center, 0.2,
							new metal(vector(0.5*(1 + RND), 0.5*(1 + RND), 0.5*(1 + RND)), 0.5*RND)));
					}
					else // glass
					{
						w->add(new sphere(center, 0.2, new dielectric(1.5)));
					}
				}
			}
		}

		w->add(new sphere(vector(0, 1, 0), 1.0, new dielectric(1.5)));
		w->add(new sphere(vector(-4, 1, 0), 1.0, new lambertian(vector(0.4, 0.2, 0.1))));
		w->add(new sphere(vector(4, 1, 0), 1.0, new metal(vector(0.7, 0.6, 0.5), 0.0)));
//		w->add(new sphere(vector(0, 0, -1), 0.5, new lambertian(vector(0.1, 0.2, 0.5))));
//		w->add(new sphere(vector(0, -100.5, -1), 100, new lambertian(vector(0.8, 0.8, 0.0))));
//		w->add(new sphere(vector(1, 0, -1), 0.5, new metal(vector(0.8, 0.6, 0.2), 0.0)));

        vector lookfrom(13,2,3);
        vector lookat(0,0,-1);
        float dist_to_focus = 10.0;
        float aperture = 0.01;
		*world = w;
        *cam   = new camera(lookfrom,
                                 lookat,
                                 vector(0,1,0),
                                 20.0,
                                 float(px)/float(py),
                                 aperture,
                                 dist_to_focus);
    }
}


__global__ void freeWorld(hittable **dWorld, camera **dCamera) {
    delete *dWorld;
    delete *dCamera;
}


#define WIDTH 800
#define HEIGHT 600

int main()
{
	glfwSetErrorCallback(check_glfw);
	if (!glfwInit())
	{
		return 1;
	}

	const char* glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Ray Tracing", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		return 1;
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cerr << "Failed to initialize OpenGL loader!\n";
		glfwTerminate();
		return -1;
	}

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	(void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	// Generate a texture
	// PBO
	GLuint buffer;
	GL_CALL(glGenBuffers(1, &buffer));
	GL_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer));
	GL_CALL(glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * 3 * 4, nullptr, GL_DYNAMIC_COPY));
	// Register in CUDA
	cudaGraphicsResource* cuda_pbo_resource;
	CUDA_CALL(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, buffer, cudaGraphicsMapFlagsNone));
	GL_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));

	// Texture
	GLuint tex;
	GL_CALL(glGenTextures(1, &tex));
	GL_CALL(glBindTexture(GL_TEXTURE_2D, tex));
	GL_CALL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_FLOAT, nullptr));
	GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL_CALL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GL_CALL(glBindTexture(GL_TEXTURE_2D, 0));


	// CUDA
	int tx = 32;
	int ty = 32;
    // allocate random state
    curandState *dRandState;
    CUDA_CALL(cudaMalloc((void **)&dRandState, WIDTH*HEIGHT*sizeof(curandState)));
    curandState *dRandState2;
    CUDA_CALL(cudaMalloc((void **)&dRandState2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    randInit<<<1,1>>>(dRandState2);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    // make world
    hittable **dWorld;
    CUDA_CALL(cudaMalloc((void **)&dWorld, sizeof(hittable *)));
    camera **dCamera;
    CUDA_CALL(cudaMalloc((void **)&dCamera, sizeof(camera *)));
    createWorld<<<1,1>>>(dWorld, dCamera, WIDTH, HEIGHT, dRandState2);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

	dim3 blocks(WIDTH/tx+1,HEIGHT/ty+1);
    dim3 threads(tx,ty);

	renderInit<<<blocks, threads>>>(WIDTH, HEIGHT, dRandState);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());

	float* dptr;
	size_t num_bytes;
	int frame = 0;
	while (!glfwWindowShouldClose(window))
	{
		frame++;
		glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();


		// DockSpace
		ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

		// Viewport

		{
            ImGui::Begin("Viewport");   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Image((void*)(intptr_t)tex, ImVec2(WIDTH, HEIGHT), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::End();
        }


        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
        {
//            static float f = 0.0f;
//            static int counter = 0;

            ImGui::Begin("Settings");                          // Create a window called "Hello, world!" and append into it.

//            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
//
//            if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
//                counter++;
//            ImGui::SameLine();
//            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }

        // Update texture

		// Map buffer
		CUDA_CALL(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		CUDA_CALL(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cuda_pbo_resource));
		// Raytracing
		render<<<blocks, threads>>>(dptr, WIDTH, HEIGHT, frame, 5, dCamera, dWorld, dRandState);
		CUDA_CALL(cudaGetLastError());
		CUDA_CALL(cudaDeviceSynchronize());
		// Unmap buffer
		CUDA_CALL(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
		// Copy buffer to texture
		GL_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer));
		GL_CALL(glBindTexture(GL_TEXTURE_2D, tex));
		GL_CALL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGB, GL_FLOAT, nullptr));
		GL_CALL(glBindTexture(GL_TEXTURE_2D, 0));
		GL_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
		GL_CALL(glViewport(0, 0, display_w, display_h));
		GL_CALL(glClearColor(0.0f, 0.0f, 0.0f, 1.0f));
		GL_CALL(glClear(GL_COLOR_BUFFER_BIT));

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    CUDA_CALL(cudaDeviceSynchronize());
    freeWorld<<<1,1>>>(dWorld,dCamera);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaFree(dCamera));
    CUDA_CALL(cudaFree(dWorld));
    CUDA_CALL(cudaFree(dRandState));
    CUDA_CALL(cudaFree(dRandState2));

    cudaDeviceReset();

    return 0;

}