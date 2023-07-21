//
// Created by vigb9 on 20/07/2023.
//

#include "Raytracing/Renderer.cuh"
#include "Raytracing/Material.cuh"
#include <iostream>
#include <chrono>

#include <curand_kernel.h>

using namespace std;

__device__ Color rayColor(const Ray& r, const Hittable& world, int depth, curandState* state)
{
	HitRecord rec;

	// If we've exceeded the ray bounce limit, no more light is gathered.
	if (depth <= 0)
		return Color(0, 0, 0);

	if (world.hit(r, 0.001f, d_infinity, rec))
	{
		Ray scattered;
		Color attenuation;
		if (rec.matPtr->scatter(r, rec, attenuation, scattered, state))
			return attenuation * rayColor(scattered, world, depth - 1, state);
		return Color(0, 0, 0);
	}

	Vector unitDirection = unitVector(r.direction());
	auto t = 0.5f * (unitDirection.y() + 1.0f);
	return (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
}

__global__ void generateRay(Ray* rays, const Hittable& world, int depth,  Color* result, curandState* state)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(1234, idx, 0, &state[idx]);
	curandState localState = state[idx];
	result[idx] = rayColor(rays[idx], world, depth, &localState);
	state[idx] = localState;
}


Renderer::Renderer(int width, int height, int maxDepth, int samplesPerPixel):
	width(width), height(height), maxDepth(maxDepth), samplesPerPixel(samplesPerPixel)
{
}


std::vector<Color> Renderer::render(const Hittable& world)
{
	return std::vector<Color>();
}


vector<Color> Renderer::renderGPU(const Camera& cam, Hittable& world)
{
	world.initGPU();
	auto start = chrono::high_resolution_clock::now();
	// Generate rays
	vector<Ray> rays;
	for (int j = height - 1; j >= 0; --j)
	{
		for (int i = 0; i < width; ++i)
		{
			int idx = j * width + i;
			float u = float(i) / (width - 1);
			float v = float(j) / (height - 1);
			rays[idx] = cam.getRay(u, v);
		}
	}

	// Allocate memory on device
	Ray* d_rays;
	Color* d_result;
	curandState* state;
	cudaMalloc(&d_rays, width * height * sizeof(Ray));
	cudaMalloc(&d_result, width * height * sizeof(Color));
	cudaMalloc(&state, width * height * sizeof(curandState));

	// Copy rays to device
	world.updateGPU();
	cudaMemcpy(d_rays, rays.data(), width * height * sizeof(Ray), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	auto startup_time = chrono::high_resolution_clock::now() - start;
	cerr << "Startup time: " << chrono::duration_cast<chrono::milliseconds>(startup_time).count() << "ms" << endl;

	// Launch kernel
	start = chrono::high_resolution_clock::now();
	generateRay<<<width * height / 256 + 1, 256>>>(d_rays, world, maxDepth, d_result, state);
	cudaDeviceSynchronize();
	auto kernel_time = chrono::high_resolution_clock::now() - start;
	cerr << "Kernel execution time: " << chrono::duration_cast<chrono::milliseconds>(kernel_time).count() << "ms" << endl;

	// Copy result to host
	start = chrono::high_resolution_clock::now();
	vector<Color> result(width * height);
	cudaMemcpy(result.data(), d_result, width * height * sizeof(Color), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	auto copy_time = chrono::high_resolution_clock::now() - start;
	cerr << "Copy time: " << chrono::duration_cast<chrono::milliseconds>(copy_time).count() << "ms" << endl;
	cerr << "Total time: " << chrono::duration_cast<chrono::milliseconds>(startup_time + kernel_time + copy_time).count() << "ms" << endl;

	world.freeGPU();
	cudaFree(d_rays);
	cudaFree(d_result);
	return result;
}

Renderer::~Renderer()
{
}
