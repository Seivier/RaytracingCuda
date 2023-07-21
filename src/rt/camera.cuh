//
// Created by vigb9 on 20/07/2023.
//

#ifndef _CAMERA_CUH_
#define _CAMERA_CUH_

#include <curand_kernel.h>
#include "ray.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

CUDA_ONLY vector randomInUnitDisk(curandState* localRandState);

class camera
{
  public:
	CUDA_CALLABLE camera(vector lookFrom, vector lookAt, vector vUp, float vFov, float aspect, float aperture, float focusDist);
	CUDA_ONLY ray getRay(float s, float t, curandState* localRandState) const;

  private:
	vector origin;
	vector lowerLeftCorner;
	vector horizontal;
	vector vertical;
	vector u, v, w;
	float lensRadius;
};

#endif //_CAMERA_CUH_
