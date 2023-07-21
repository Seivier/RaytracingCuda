//
// Created by vigb9 on 20/07/2023.
//

#ifndef _SPHERE_CUH_
#define _SPHERE_CUH_

#include "hittable.cuh"

class sphere: public hittable
{
 public:
	CUDA_CALLABLE sphere() {};
	CUDA_CALLABLE sphere(point cen, float r, material* m) : center(cen), radius(r), matPtr(m) {};

	CUDA_CALLABLE bool hit(const ray& r, float tMin, float tMax, hit_record& rec) const override;


 public:
	material* matPtr;

 private:
	point center;
	float radius;
};

#endif //_SPHERE_CUH_
