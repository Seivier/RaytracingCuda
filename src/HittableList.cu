//
// Created by vigb9 on 19/07/2023.
//

#include "Raytracing/HittableList.cuh"
bool HittableList::hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const
{
	HitRecord tempRec;
	bool hitAnything = false;
	auto closestSoFar = tMax;

	for (const auto& object : objects)
	{
		if (object->hit(r, tMin, closestSoFar, tempRec))
		{
			hitAnything = true;
			closestSoFar = tempRec.t;
			rec = tempRec;
		}
	}

	return hitAnything;
}
