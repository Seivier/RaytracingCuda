//
// Created by vigb9 on 19/07/2023.
//

#include "Raytracing/HittableList.cuh"

__device__ bool HittableList::hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const
{
	HitRecord tempRec;
	bool hitAnything = false;
	auto closestSoFar = tMax;

	for (int i = 0; i < count; i++)
	{
		auto object = d_objects[i];
		if (object->hit(r, tMin, closestSoFar, tempRec))
		{
			hitAnything = true;
			closestSoFar = tempRec.t;
			rec = tempRec;
		}
	}

	return hitAnything;
}
void HittableList::initGPU()
{
	cudaMalloc(&d_objects, objects.size() * sizeof(Hittable*));
	for (int i = 0; i < objects.size(); i++)
	{
		objects[i]->initGPU();
	}
	count = objects.size();
}


void HittableList::freeGPU()
{
	cudaFree(d_objects);
	for (auto& object : objects)
	{
		object->freeGPU();
	}
	count = 0;
}

void HittableList::updateGPU()
{
	cudaMemcpy(d_objects, objects.data(), objects.size() * sizeof(Hittable*), cudaMemcpyHostToDevice);
	for (int i = 0; i < objects.size(); i++)
	{
		objects[i]->updateGPU();
		cudaMemcpy(&d_objects[i], &objects[i], sizeof(Hittable*), cudaMemcpyHostToDevice);
	}
	count = objects.size();
}
