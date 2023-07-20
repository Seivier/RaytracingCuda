//
// Created by vigb9 on 19/07/2023.
//

#ifndef _SPHERE_CUH_
#define _SPHERE_CUH_

#include "Raytracing/Hittable.cuh"

class Sphere: public Hittable
{
 public:
	Sphere() = default;
	Sphere(const Point& center, float radius, shared_ptr<Material> m) : center(center), radius(radius), matPtr(m) {}
	bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const override;

 private:
	Point center;
	float radius;
	shared_ptr<Material> matPtr;
};



#endif //_SPHERE_CUH_
