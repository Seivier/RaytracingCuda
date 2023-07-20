//
// Created by vigb9 on 20/07/2023.
//

#include "Raytracing/Material.cuh"
#include "Raytracing/Hittable.cuh"

bool Lambertian::scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const
{
	auto scatter_direction = rec.normal + randomUnitVector();
	// Catch degenerate scatter direction
	if (scatter_direction.nearZero())
		scatter_direction = rec.normal;
	scattered = Ray(rec.p, scatter_direction);
	attenuation = albedo;
	return true;
}

bool Metal::scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const
{
	Vector reflected = reflect(unitVector(r_in.direction()), rec.normal);
	scattered = Ray(rec.p, reflected + fuzz*randomInUnitSphere());
	attenuation = albedo;
	return (dot(scattered.direction(), rec.normal) > 0);
}

bool Dielectric::scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const
{
	attenuation = Color(1.0f, 1.0f, 1.0f);
	float refraction_ratio = rec.frontFace ? (1.0f / ir) : ir;

	Vector unit_direction = unitVector(r_in.direction());
	float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0f);
	float sin_theta = sqrt(1.0f - cos_theta*cos_theta);

	bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
	Vector direction;

	if (cannot_refract || reflectance(cos_theta, refraction_ratio) > randomFloat())
		direction = reflect(unit_direction, rec.normal);
	else
		direction = refract(unit_direction, rec.normal, refraction_ratio);

	scattered = Ray(rec.p, direction);
	return true;
}

float Dielectric::reflectance(float cosine, float ref_idx)
{
	// Use Schlick's approximation for reflectance.
	auto r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0*r0;
	return r0 + (1 - r0)*pow((1 - cosine), 5);
}
