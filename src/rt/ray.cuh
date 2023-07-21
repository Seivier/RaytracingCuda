//
// Created by vigb9 on 20/07/2023.
//

#ifndef _RAY_CUH_
#define _RAY_CUH_

#include "vector.cuh"

class ray
{
 public:
	CUDA_CALLABLE ray() {};
	CUDA_CALLABLE ray(const point& origin, const vector& direction): orig(origin), dir(direction) {};
	CUDA_CALLABLE vector origin() const;
	CUDA_CALLABLE vector direction() const;
	CUDA_CALLABLE point at(float t) const;
 private:
	point orig;
	vector dir;
};

#endif //_RAY_CUH_
