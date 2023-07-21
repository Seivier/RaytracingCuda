//
// Created by vigb9 on 20/07/2023.
//

#include "sphere.cuh"

CUDA_CALLABLE bool sphere::hit(const ray& r, float tMin, float tMax, hit_record& rec) const
{
	vector oc = r.origin() - center;
	float a = r.direction().squaredLength();
	float halfB = dot(oc, r.direction());
	float c = oc.squaredLength() - radius * radius;
	float discriminant = halfB * halfB - a * c;

	if (discriminant > 0)
	{
		float root = sqrt(discriminant);
		float temp = (-halfB - root) / a;
		if (temp < tMax && temp > tMin)
		{
			rec.t = temp;
			rec.p = r.at(rec.t);
			vector outwardNormal = (rec.p - center) / radius;
			rec.setFaceNormal(r, outwardNormal);
			rec.matPtr = matPtr;
			return true;
		}
		temp = (-halfB + root) / a;
		if (temp < tMax && temp > tMin)
		{
			rec.t = temp;
			rec.p = r.at(rec.t);
			vector outwardNormal = (rec.p - center) / radius;
			rec.setFaceNormal(r, outwardNormal);
			rec.matPtr = matPtr;
			return true;
		}
	}
	return false;
}