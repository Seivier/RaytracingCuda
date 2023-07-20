//
// Created by vigb9 on 20/07/2023.
//

#ifndef _MATERIAL_CUH_
#define _MATERIAL_CUH_

#include "Raytracing/Utils.cuh"
#include "Raytracing/Color.cuh"

struct HitRecord;

class Material
{
 public:
	virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const = 0;
};

class Lambertian : public Material
{
 public:
	Lambertian(const Color& a)
		: albedo(a)
	{
	}

	bool scatter(
		const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered
	) const override;

 private:
	Color albedo;
};

class Metal : public Material
{
 public:
	Metal(const Color& a, float f)
		: albedo(a), fuzz(f < 1 ? f : 1)
	{
	}

	virtual bool scatter(
		const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered
	) const override;

 public:
	Color albedo;
	float fuzz;
};

class Dielectric : public Material
{
 public:
	Dielectric(float index_of_refraction)
		: ir(index_of_refraction)
	{
	}

	bool scatter(
		const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered
	) const override;

 private:
	float ir; // Index of Refraction

	static float reflectance(float cosine, float ref_idx);

};

#endif //_MATERIAL_CUH_
