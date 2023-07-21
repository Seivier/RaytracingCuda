//
// Created by vigb9 on 20/07/2023.
//
#include "ray.cuh"

__host__ __device__ vector ray::origin() const
{
	return orig;
}
__host__ __device__ vector ray::direction() const
{
	return dir;
}
__host__ __device__ point ray::at(float t) const
{
	return orig + t * dir;
}
