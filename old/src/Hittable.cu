//
// Created by vigb9 on 19/07/2023.
//

#include "Raytracing/Hittable.cuh"
#include "Raytracing/Material.cuh"

__device__ void HitRecord::setFaceNormal(const Ray& r, const Vector& outwardNormal)
{
	frontFace = dot(r.direction(), outwardNormal) < 0;
	normal = frontFace ? outwardNormal : -outwardNormal;
}