//
// Created by vigb9 on 19/07/2023.
//

#ifndef _SPHERE_CUH_
#define _SPHERE_CUH_

#include "Raytracing/Hittable.cuh"

#define CUDA_CALLABLE_MEMBER __host__ __device__

class Sphere: public Hittable
{
 public:
	Sphere() = default;
	Sphere(const Point& center, float radius, shared_ptr<Material>& m) : center(center), radius(radius), matPtr(m) {}
	__device__ bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override;

	void initGPU() override;
	void updateGPU() override;
	void freeGPU() override;

 private:
	Point center;
	float radius;
	shared_ptr<Material> matPtr;
	Material* d_matPtr;
};



#endif //_SPHERE_CUH_
