//
// Created by vigb9 on 20/07/2023.
//

#include "hittable.cuh"

CUDA_CALLABLE void hit_record::setFaceNormal(const ray& r, const vector& outwardNormal)
{
	frontFace = dot(r.direction(), outwardNormal) < 0;
	normal = frontFace ? outwardNormal : -outwardNormal;
}