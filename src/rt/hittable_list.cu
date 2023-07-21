//
// Created by vigb9 on 20/07/2023.
//

#include "hittable_list.cuh"

__host__ __device__ hittable_list::hittable_list()
{
	capacity = 10;
	count = 0;
	list = new hittable * [capacity];
}
__host__ __device__ hittable_list::hittable_list(hittable** l, int n)
{
	capacity = n;
	count = n;
	list = new hittable * [capacity];
	for (int i = 0; i < n; i++)
	{
		list[i] = l[i];
	}
}

__host__ __device__ bool hittable_list::hit(const ray& r, float tMin, float tMax, hit_record& rec) const
{
	hit_record tempRec;
	bool hitAnything = false;
	float closestSoFar = tMax;
	for (int i = 0; i < count; i++)
	{
		if (list[i]->hit(r, tMin, closestSoFar, tempRec))
		{
			hitAnything = true;
			closestSoFar = tempRec.t;
			rec = tempRec;
		}
	}
	return hitAnything;
}
__host__ __device__ void hittable_list::add(hittable* obj)
{
	if (count == capacity)
	{
		capacity *= 2;
		auto** newList = new hittable * [capacity];
		for (int i = 0; i < count; i++)
		{
			newList[i] = list[i];
		}
		delete[] list;
		list = newList;
	}
	list[count++] = obj;
}
__host__ __device__ hittable_list::~hittable_list()
{
	for (int i = 0; i < count; i++)
	{
		delete list[i];
	}
	delete[] list;
}


