//
// Created by vigb9 on 19/07/2023.
//

#ifndef _HITTABLELIST_CUH_
#define _HITTABLELIST_CUH_

#include <memory>
#include <vector>

#include "Raytracing/Hittable.cuh"

#define CUDA_CALLABLE_MEMBER __host__ __device__

class HittableList: public Hittable
{
 public:
	HittableList() = default;
	HittableList(std::shared_ptr<Hittable> object) { add(object); }

	inline void clear() { objects.clear(); }
	inline void add(std::shared_ptr<Hittable> object) { objects.push_back(object); }

	void initGPU() override;
	void updateGPU() override;
	void freeGPU() override;

	__device__ bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override;


 private:
	std::vector<std::shared_ptr<Hittable>> objects;
	Hittable** d_objects;
	size_t count;

};

#endif //_HITTABLELIST_CUH_
