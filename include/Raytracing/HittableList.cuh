//
// Created by vigb9 on 19/07/2023.
//

#ifndef _HITTABLELIST_CUH_
#define _HITTABLELIST_CUH_

#include <memory>
#include <vector>

#include "Raytracing/Hittable.cuh"

class HittableList: public Hittable
{
 public:
	HittableList() = default;
	HittableList(std::shared_ptr<Hittable> object) { add(object); }

	inline void clear() { objects.clear(); }
	inline void add(std::shared_ptr<Hittable> object) { objects.push_back(object); }

	bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override;

 private:
	std::vector<std::shared_ptr<Hittable>> objects;

};

#endif //_HITTABLELIST_CUH_
