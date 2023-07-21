//
// Created by vigb9 on 20/07/2023.
//

#include "material.cuh"
#define RANDVECTOR vector(curand_uniform(localRandState), curand_uniform(localRandState), curand_uniform(localRandState))

__host__ __device__ float
schlick(float cosine, float refIdx)
{
	float r0 = (1.0f - refIdx) / (1.0f + refIdx);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}
__host__ __device__ bool
refract(const vector& v, const vector& n, float niOverNt, vector& refracted)
{
	vector uv = unitVector(v);
	float dt = dot(uv, n);
	float discriminant = 1.0f - niOverNt * niOverNt * (1.0f - dt * dt);
	if (discriminant > 0.0f)
	{
		refracted = niOverNt * (uv - n * dt) - n * sqrtf(discriminant);
		return true;
	}
	else
		return false;
}
__host__ __device__ vector
reflect(const vector& v, const vector& n)
{
	return v - 2.0f * dot(v, n) * n;
}
__device__ vector
randomInUnitSphere(curandState* localRandState)
{
	vector p;
	do
	{
		p = 2.0f * RANDVECTOR - vector(1.0f, 1.0f, 1.0f);
	} while (p.squaredLength() >= 1.0f);
	return p;
}
__device__ bool
lambertian::scatter(const ray& rIn, const hit_record& rec, vector& attenuation, ray& scattered, curandState* localRandState) const
{
	vector target = rec.p + rec.normal + randomInUnitSphere(localRandState);
	scattered = ray(rec.p, target - rec.p);
	attenuation = albedo;
	return true;
}
__device__ bool
metal::scatter(const ray& rIn, const hit_record& rec, vector& attenuation, ray& scattered, curandState* localRandState) const
{
	vector reflected = reflect(unitVector(rIn.direction()), rec.normal);
	scattered = ray(rec.p, reflected + fuzz * randomInUnitSphere(localRandState));
	attenuation = albedo;
	return (dot(scattered.direction(), rec.normal) > 0.0f);
}
__device__ bool
dielectric::scatter(const ray& rIn, const hit_record& rec, vector& attenuation, ray& scattered, curandState* localRandState) const
{
	vector outwardNormal;
	vector reflected = reflect(rIn.direction(), rec.normal);
	float niOverNt;
	attenuation = vector(1.0f, 1.0f, 1.0f);
	vector refracted;
	float reflectProb;
	float cosine;
	if (dot(rIn.direction(), rec.normal) > 0.0f)
	{
		outwardNormal = -rec.normal;
		niOverNt = refIdx;
		cosine = refIdx * dot(rIn.direction(), rec.normal) / rIn.direction().length();
	}
	else
	{
		outwardNormal = rec.normal;
		niOverNt = 1.0f / refIdx;
		cosine = -dot(rIn.direction(), rec.normal) / rIn.direction().length();
	}

	if (refract(rIn.direction(), outwardNormal, niOverNt, refracted))
		reflectProb = schlick(cosine, refIdx);
	else
		reflectProb = 1.0f;

	if (curand_uniform(localRandState) < reflectProb)
		scattered = ray(rec.p, reflected);
	else
		scattered = ray(rec.p, refracted);
	return true;
}
