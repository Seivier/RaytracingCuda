//
// Created by vigb9 on 19/07/2023.
//

#ifndef _HITTABLE_CUH_
#define _HITTABLE_CUH_

#include "Raytracing/Ray.cuh"
#include "Raytracing/Utils.cuh"

class Material;

struct HitRecord
{
	Point p;
	Vector normal;
	shared_ptr<Material> matPtr;
	float t;
	bool frontFace;

	void setFaceNormal(const Ray& r, const Vector& outwardNormal);
};

class Hittable
{
 public:
	virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const = 0;
};

#endif //_HITTABLE_CUH_
