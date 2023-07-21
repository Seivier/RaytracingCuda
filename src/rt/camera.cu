//
// Created by vigb9 on 20/07/2023.
//

#include "camera.cuh"
__device__ vector randomInUnitDisk(curandState* localRandState)
{
	vector p;
	do
	{
		p = 2.0f * vector(curand_uniform(localRandState), curand_uniform(localRandState), 0) - vector(1, 1, 0);
	} while (dot(p, p) >= 1.0f);
	return p;
}
__host__ __device__ camera::camera(vector lookFrom, vector lookAt, vector vUp, float vFov, float aspect, float aperture, float focusDist)
{
	lensRadius = aperture / 2.0f;
	float theta = vFov * ((float)M_PI) / 180.0f;
	float halfHeight = tanf(theta / 2.0f);
	float halfWidth = aspect * halfHeight;
	origin = lookFrom;
	w = unitVector(lookFrom - lookAt);
	u = unitVector(cross(vUp, w));
	v = cross(w, u);
	lowerLeftCorner = origin - halfWidth * focusDist * u - halfHeight * focusDist * v - focusDist * w;
	horizontal = 2.0f * halfWidth * focusDist * u;
	vertical = 2.0f * halfHeight * focusDist * v;
}
__device__ ray camera::getRay(float s, float t, curandState* localRandState) const
{
	vector rd = lensRadius * randomInUnitDisk(localRandState);
	vector offset = u * rd.x() + v * rd.y();
	return ray(origin + offset, lowerLeftCorner + s * horizontal + t * vertical - origin - offset);
}

