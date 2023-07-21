//
// Created by vigb9 on 20/07/2023.
//

#include "hittable_list.cuh"
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
