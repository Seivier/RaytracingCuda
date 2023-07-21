//
// Created by vigb9 on 20/07/2023.
//

#ifndef _HITTABLE_CUH_
#define _HITTABLE_CUH_

#include "ray.cuh"

class material;

struct hit_record
{
	point p;
	vector normal;
	float t;
	bool frontFace;
	material* matPtr;

	CUDA_CALLABLE void setFaceNormal(const ray& r, const vector& outwardNormal);
};

class hittable
{
 public:
	CUDA_CALLABLE virtual ~hittable() {};
	CUDA_CALLABLE virtual bool hit(const ray& r, float tMin, float tMax, hit_record& rec) const = 0;
};

#endif //_HITTABLE_CUH_
