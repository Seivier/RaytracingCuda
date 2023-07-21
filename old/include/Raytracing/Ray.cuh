//
// Created by vigb9 on 19/07/2023.
//

#ifndef _RAY_CUH_
#define _RAY_CUH_

#include "Raytracing/Vector.cuh"

class Ray
{
 public:
	__host__ __device__ Ray(): orig(), dir() {};
	__host__ __device__ Ray(const Point& origin, const Vector& direction) : orig(origin), dir(direction) {}

	__host__ __device__ inline Point origin() const { return orig; }
	__host__ __device__ inline Vector direction() const { return dir; }

	__host__ __device__ Point at(float t) const { return orig + t*dir; }


 private:
	Point orig;
	Vector dir;
};



#endif //_RAY_CUH_
