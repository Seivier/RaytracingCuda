#include <iostream>
#include <chrono>
#include <curand_kernel.h>
#include "rt/vector.cuh"
#include "rt/ray.cuh"
#include "rt/sphere.cuh"
#include "rt/hittable_list.cuh"
#include "rt/camera.cuh"
#include "rt/material.cuh"


#define CUDA_CALL(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into rayColor() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vector rayColor(const ray& r, hittable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vector cur_attenuation = vector(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vector attenuation;
            if(rec.matPtr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vector(0.0,0.0,0.0);
            }
        }
        else {
            vector unit_direction = unitVector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vector c = (1.0f-t)*vector(1.0, 1.0, 1.0) + t*vector(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vector(0.0,0.0,0.0); // exceeded recursion
}

__global__ void randInit(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void renderInit(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(color *fb, int max_x, int max_y, int ns, camera **cam, hittable **world, curandState *randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = randState[pixel_index];
    color col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->getRay(u, v, &local_rand_state);
        col += rayColor(r, world, &local_rand_state);
    }
    randState[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void createWorld(hittable **world, camera **cam, int px, int py, curandState *randState) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState localRand = *randState;
        auto w = new hittable_list();
		w->add(new sphere(vector(0,-1000,0), 1000, new lambertian(vector(0.1, 0.5, 0.1))));
		w->add(new sphere(vector(0, 1, 0), 1.0, new dielectric(1.5)));
		w->add(new sphere(vector(-4, 1, 0), 1.0, new lambertian(vector(0.4, 0.2, 0.1))));
        vector lookfrom(0,1,13);
        vector lookat(0,0,0);
        float dist_to_focus = (lookfrom-lookat).length();
        float aperture = 0.1;
		*world = w;
        *cam   = new camera(lookfrom,
                                 lookat,
                                 vector(0,1,0),
                                 30.0,
                                 float(px)/float(py),
                                 aperture,
                                 dist_to_focus);
    }
}


__global__ void freeWorld(hittable **dWorld, camera **dCamera) {
    delete *dWorld;
    delete *dCamera;
}

int main(int argc, char** argv) {

	if (argc != 6)
	{
		std::cerr << "Usage: " << argv[0] << " <width> <height> <block_size_x> <block_size_y> <sample per pixel>\n";
		return 1;
	}

    int px = atoi(argv[1]);
    int py = atoi(argv[2]);
    int tx = atoi(argv[3]);
    int ty = atoi(argv[4]);
    int ns = atoi(argv[5]);

    int numPixels = px*py;


    color *pixels;
    CUDA_CALL(cudaMallocManaged((void **)&pixels, numPixels*sizeof(color)));

    // allocate random state
    curandState *dRandState;
    CUDA_CALL(cudaMalloc((void **)&dRandState, numPixels*sizeof(curandState)));
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
    createWorld<<<1,1>>>(dWorld, dCamera, px, py, dRandState);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    // Render our buffer
    dim3 blocks(px/tx+1,py/ty+1);
    dim3 threads(tx,ty);
    renderInit<<<blocks, threads>>>(px, py, dRandState);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(pixels, px, py,  ns, dCamera, dWorld, dRandState);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();
	auto timer_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(stop-start);
	std::cerr << "Render: took " << timer_seconds.count() << " seconds.\n";

    // Output FB as Image
	start = std::chrono::high_resolution_clock::now();
    std::cout << "P3\n" << px << " " << py << "\n255\n";
    for (int j = py-1; j >= 0; j--) {
        for (int i = 0; i < px; i++) {
            size_t pixel_index = j*px + i;
            int ir = int(255.99*pixels[pixel_index].r());
            int ig = int(255.99*pixels[pixel_index].g());
            int ib = int(255.99*pixels[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
	stop = std::chrono::high_resolution_clock::now();
	timer_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(stop-start);
	std::cerr << "Output: took " << timer_seconds.count() << " seconds.\n";

    // clean up
    CUDA_CALL(cudaDeviceSynchronize());
    freeWorld<<<1,1>>>(dWorld,dCamera);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaFree(dCamera));
    CUDA_CALL(cudaFree(dWorld));
    CUDA_CALL(cudaFree(dRandState));
    CUDA_CALL(cudaFree(dRandState2));
    CUDA_CALL(cudaFree(pixels));

    cudaDeviceReset();
}