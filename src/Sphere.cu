//
// Created by vigb9 on 19/07/2023.
//

#include "Raytracing/Sphere.cuh"
#include <glm/gtx/norm.hpp>

bool Sphere::hit(const Ray& r, float tMin, float tMax, HitRecord& rec) const
{
	Vector oc = r.origin() - center;
	auto a = r.direction().squaredLength();
	auto half_b = dot(oc, r.direction());
	auto c = oc.squaredLength() - radius * radius;

	auto discriminant = half_b * half_b - a * c;
	if (discriminant < 0) return false;
	auto sqrtd = sqrt(discriminant);

	// Find the nearest root that lies in the acceptable range.
	auto root = (-half_b - sqrtd) / a;
	if (root < tMin || tMax < root)
	{
		root = (-half_b + sqrtd) / a;
		if (root < tMin || tMax < root)
			return false;
	}

	rec.t = root;
	rec.p = r.at(rec.t);
	Vector outwardNormal = (rec.p - center) / radius;
	rec.setFaceNormal(r, outwardNormal);
	rec.matPtr = matPtr;

	return true;
}

