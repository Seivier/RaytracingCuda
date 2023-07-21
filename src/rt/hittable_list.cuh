//
// Created by vigb9 on 20/07/2023.
//

#ifndef _HITTABLE_LIST_CUH_
#define _HITTABLE_LIST_CUH_

#include "hittable.cuh"

class hittable_list: public hittable
{
 public:
	CUDA_CALLABLE hittable_list();
	CUDA_CALLABLE hittable_list(hittable** l, int n);
	CUDA_CALLABLE ~hittable_list() override;
	CUDA_CALLABLE void add(hittable* obj);
	CUDA_CALLABLE bool hit(const ray& r, float tMin, float tMax, hit_record& rec) const override;
	CUDA_CALLABLE inline hittable* operator[](int i) const { return list[i]; };
	CUDA_CALLABLE inline hittable*& operator[](int i) { return list[i]; };
 public:
	int count;
 private:
	hittable** list;
	int capacity;
};

#endif //_HITTABLE_LIST_CUH_
