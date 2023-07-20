//
// Created by vigb9 on 19/07/2023.
//

#ifndef _RAY_CUH_
#define _RAY_CUH_

#include "Raytracing/Vector.cuh"

class Ray
{
 public:
	Ray() = default;
	Ray(const Point& origin, const Vector& direction) : orig(origin), dir(direction) {}

	inline Point origin() const { return orig; }
	inline Vector direction() const { return dir; }

	Point at(float t) const { return orig + t*dir; }


 private:
	Point orig;
	Vector dir;
};



#endif //_RAY_CUH_
