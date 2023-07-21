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
	Material* matPtr;
	float t;
	bool frontFace;

	__device__ HitRecord(): p(), normal(), t(), frontFace() {};

	__device__ void setFaceNormal(const Ray& r, const Vector& outwardNormal);

	// copy
	__device__ HitRecord(const HitRecord& other) : p(other.p), normal(other.normal), t(other.t), matPtr(other.matPtr), frontFace(other.frontFace) {};
	__device__ HitRecord& operator=(const HitRecord& other) { p = other.p; normal = other.normal; t = other.t; matPtr = other.matPtr; frontFace = other.frontFace; return *this; };

};

class Hittable
{
 public:
	virtual void initGPU() = 0;
	virtual void updateGPU() = 0;
	virtual void freeGPU() = 0;

	__device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const = 0;
};

#endif //_HITTABLE_CUH_
