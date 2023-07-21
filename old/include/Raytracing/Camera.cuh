//
// Created by vigb9 on 19/07/2023.
//

#ifndef _CAMERA_CUH_
#define _CAMERA_CUH_

#include "Raytracing/Utils.cuh"

class Camera
{
    public:
        Camera(Point lookFrom, Point lookAt, Vector vup, float vfov, float aspect_ratio, float aperture, float focusDist);
		Ray getRay(float s, float t) const;

    private:
        Point origin;
        Point lower_left_corner;
        Vector horizontal;
        Vector vertical;
		Vector u, v, w;
		float lensRadius;

};

#endif //_CAMERA_CUH_
