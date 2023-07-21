//
// Created by vigb9 on 20/07/2023.
//
#include <iostream>
#include <chrono>
#include <vector>

#include "Raytracing/Utils.cuh"
#include "Raytracing/HittableList.cuh"
#include "Raytracing/Sphere.cuh"
#include "Raytracing/Camera.cuh"
#include "Raytracing/Material.cuh"
#include "Raytracing/Renderer.cuh"


int main()
{
	// Image
	const float aspect_ratio = 16.0f / 9.0f;
	const int image_width = 400;
	const int image_height = static_cast<int>(image_width / aspect_ratio);
	const int max_depth = 50;
//	const int samples_per_pixel = 100;

	// World
	HittableList world;

	auto material_ground = make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
	auto material_left = make_shared<Dielectric>(1.5);
    auto material_center = make_shared<Lambertian>(Color(0.1, 0.2, 0.5));
    auto material_right  = make_shared<Metal>(Color(0.8, 0.6, 0.2), 0.0);

	world.add(make_shared<Sphere>(Point( 0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(make_shared<Sphere>(Point( 0.0,    0.0, -1.0),   0.5, material_center));
    world.add(make_shared<Sphere>(Point(-1.0,    0.0, -1.0),   0.5, material_left));
    world.add(make_shared<Sphere>(Point(-1.0,    0.0, -1.0),   -0.45, material_left));
    world.add(make_shared<Sphere>(Point( 1.0,    0.0, -1.0),   0.5, material_right));

	Point lookFrom(3, 3, 2);
	Point lookAt(0, 0, -1);
	Vector vup(0, 1, 0);

	float dist_to_focus = (lookFrom - lookAt).length();
	float aperture = 2.0f;

	// Camera
	Camera cam(lookFrom, lookAt, vup, 20, aspect_ratio, aperture, dist_to_focus);

	// Render
	Renderer renderer(image_width, image_height, max_depth, 1);

	auto result = renderer.renderGPU(cam, world);

	// Write to file
	std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
	for (int j = image_height - 1; j >= 0; --j)
	{
		for (int i = 0; i < image_width; ++i)
		{
			int idx = j * image_width + i;
			writeColor(std::cout, result[idx], 1);
		}
	}
}