//
// Created by vigb9 on 20/07/2023.
//

#ifndef _MATERIAL_CUH_
#define _MATERIAL_CUH_

#include "ray.cuh"
#include "hittable.cuh"

#include <curand_kernel.h>


CUDA_CALLABLE float schlick(float cosine, float refIdx);
CUDA_CALLABLE bool refract(const vector& v, const vector& n, float niOverNt, vector& refracted);
CUDA_CALLABLE vector reflect(const vector& v, const vector& n);

CUDA_ONLY vector randomInUnitSphere(curandState* localRandState);

class material
{
 public:
	CUDA_ONLY virtual bool scatter(const ray& rIn, const hit_record& rec, vector& attenuation, ray& scattered, curandState* localRandState) const = 0;
};


class lambertian : public material
{
 public:
	CUDA_CALLABLE lambertian(const vector& a): albedo(a) {}
	CUDA_ONLY bool scatter(const ray& rIn, const hit_record& rec, vector& attenuation, ray& scattered, curandState* localRandState) const override;

 private:
	color albedo;
};

class metal : public material
{
 public:
	CUDA_CALLABLE metal(const vector& a, float f): albedo(a), fuzz(f < 1 ? f : 1) {}
	CUDA_ONLY bool scatter(const ray& rIn, const hit_record& rec, vector& attenuation, ray& scattered, curandState* localRandState) const override;

 private:
	color albedo;
	float fuzz;
};

class dielectric: public material
{
 public:
	CUDA_CALLABLE dielectric(float ri): refIdx(ri) {}
	CUDA_ONLY bool scatter(const ray& rIn, const hit_record& rec, vector& attenuation, ray& scattered, curandState* localRandState) const override;

 private:
	float refIdx;
};


#endif //_MATERIAL_CUH_
