//
// Created by vigb9 on 20/07/2023.
//

#ifndef _HITTABLE_LIST_CUH_
#define _HITTABLE_LIST_CUH_

#include "hittable.cuh"

class hittable_list: public hittable
{
 public:
	CUDA_CALLABLE hittable_list() {};
	CUDA_CALLABLE hittable_list(hittable** l, int n) { list = l; count = n; };
	CUDA_CALLABLE bool hit(const ray& r, float tMin, float tMax, hit_record& rec) const override;

 private:
	hittable** list;
	int count;
};

#endif //_HITTABLE_LIST_CUH_
