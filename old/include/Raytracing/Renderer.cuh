//
// Created by vigb9 on 20/07/2023.
//

#ifndef _RENDERER_CUH_
#define _RENDERER_CUH_

#include <vector>

#include "Raytracing/Utils.cuh"
#include "Raytracing/Camera.cuh"
#include "Raytracing/HittableList.cuh"


class Renderer
{
 public:
	Renderer(int width, int height, int maxDepth, int samplesPerPixel);
	~Renderer();

	std::vector<Color> render(const Hittable& world);
	std::vector<Color> renderGPU(const Camera& cam, Hittable& world);

 private:


	int width, height;
	int maxDepth;
	int samplesPerPixel;
};

#endif //_RENDERER_CUH_
