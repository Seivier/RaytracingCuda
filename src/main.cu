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
        cudaDeviceReset();
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

__global__ void render(color *pixels, int px, int py, int ns, int depth, camera **cam, hittable **world, curandState *randState) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= px) || (j >= py)) return;
    int pIdx = j*px + i;
    curandState localRand = randState[pIdx];
    color col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&localRand)) / float(px);
        float v = float(j + curand_uniform(&localRand)) / float(py);
        ray r = (*cam)->getRay(u, v, &localRand);
        col += rayColor(r, world, depth, &localRand);
    }
    randState[pIdx] = localRand;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    pixels[pIdx] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void createWorld(hittable **world, camera **cam, int px, int py, curandState *randState) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState localRand = *randState;
        auto w = new hittable_list();
		w->add(new sphere(vector(0,-100.5,-1), 100, new lambertian(vector(0.8, 0.8, 0.0))));
		w->add(new sphere(vector(0,0.0,-1), 0.5, new lambertian(vector(0.1, 0.2, 0.5))));
		w->add(new sphere(vector(-1,0.0,-1), 0.5, new dielectric(1.5)));
		w->add(new sphere(vector(-1,0.0,-1), -0.45, new dielectric(1.5)));
		w->add(new sphere(vector(1,0.0,-1), 0.5, new metal(vector(0.8, 0.6, 0.2), 0.0)));
        vector lookfrom(-2,2,1);
        vector lookat(0,0,-1);
        float dist_to_focus = (lookfrom-lookat).length();
        float aperture = 0.01;
		*world = w;
        *cam   = new camera(lookfrom,
                                 lookat,
                                 vector(0,1,0),
                                 90.0,
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

	if (argc != 7)
	{
		std::cerr << "Usage: " << argv[0] << " <width> <height> <block_size_x> <block_size_y> <sample per pixel> <ray lifetime>\n";
		return 1;
	}

    int px = atoi(argv[1]);
    int py = atoi(argv[2]);
    int tx = atoi(argv[3]);
    int ty = atoi(argv[4]);
    int ns = atoi(argv[5]);
	int depth = atoi(argv[6]);

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
    render<<<blocks, threads>>>(pixels, px, py, ns, depth, dCamera, dWorld, dRandState);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();
	auto timer_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(stop-start);
	std::cerr << "Render: took " << timer_seconds.count() << " seconds.\n";

    // Output FB as Image
	start = std::chrono::high_resolution_clock::now();
    std::cout << "P3\n" << px << " " << py << "\n255\n";
    for (int j = py-1; j >= 0; j--) {
		std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
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
	std::cerr << "\nOutput: took " << timer_seconds.count() << " seconds.\n";

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