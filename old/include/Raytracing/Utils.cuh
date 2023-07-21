//
// Created by vigb9 on 19/07/2023.
//

#ifndef _UTILS_CUH_
#define _UTILS_CUH_

#include <cmath>
#include <limits>
#include <memory>

#include <curand_kernel.h>

#define CUDA_CALLABLE_MEMBER __host__ __device__

// Usings

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants

const float infinity = std::numeric_limits<float>::infinity();
__device__ float d_infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385f;
__device__ float pi_d = 3.1415926535897932385f;

// Utility Functions

//__device__ inline float degreesToRadians(float degrees)
//{
//	return degrees * pi_d / 180.0f;
//}

inline float degreesToRadians(float degrees)
{
	return degrees * pi / 180.0f;
}

inline float randomFloat() {
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0f);
}

__device__ inline float randomFloat(curandState* state) {
	return curand_uniform(state);
}

inline float randomFloat(float min, float max) {
    // Returns a random real in [min,max).
    return min + (max-min) * randomFloat();
}

__device__ inline float randomFloat(float min, float max, curandState* state) {
	return min + (max - min) * randomFloat(state);
}

CUDA_CALLABLE_MEMBER inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

#include "Raytracing/Ray.cuh"
#include "Raytracing/Vector.cuh"

#endif //_UTILS_CUH_
