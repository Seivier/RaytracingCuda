 //
// Created by vigb9 on 19/07/2023.
//

#include "Raytracing/Camera.cuh"


Camera::Camera(Point lookFrom, Point lookAt, Vector vup, float vfov, float aspect_ratio, float aperture, float focusDist)
{
	auto theta = degreesToRadians(vfov);
	auto h = tan(theta / 2);
	auto viewport_height = 2.0f*h;
	auto viewport_width = aspect_ratio * viewport_height;

	w = unitVector(lookFrom - lookAt);
	u = unitVector(cross(vup, w));
	v = cross(w, u);

	origin = lookFrom;
	horizontal = focusDist * viewport_width * u;
	vertical = focusDist * viewport_height * v;
	lower_left_corner = origin - horizontal / 2 - vertical / 2 - focusDist*w;

	lensRadius = aperture / 2;
}

Ray Camera::getRay(float s, float t) const
{
	Vector rd = lensRadius * randomInUnitDisk();
	Vector offset = u * rd.x() + v * rd.y();

	return Ray(
		origin + offset,
		lower_left_corner + s*horizontal + t*vertical - origin - offset
	);
}