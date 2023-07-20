#include <iostream>

#include <Raytracing/Utils.cuh>

#include <Raytracing/HittableList.cuh>
#include <Raytracing/Sphere.cuh>
#include <Raytracing/Camera.cuh>
#include <Raytracing/Material.cuh>

#include <glm/glm.hpp>


Color rayColor(const Ray&r, const Hittable& world, int depth)
{
	HitRecord rec;

	if (depth <= 0)
		return Color(0.0f, 0.0f, 0.0f);

	if (world.hit(r, 0.001f, infinity, rec))
	{
		//Point target = rec.p + rec.normal + randomUnitVector();
		//return 0.5f * rayColor(Ray(rec.p, target - rec.p), world, depth-1);
		Ray scattered;
        Color attenuation;
        if (rec.matPtr->scatter(r, rec, attenuation, scattered))
            return attenuation * rayColor(scattered, world, depth-1);
        return Color(0,0,0);
	}
	Vector unit_direction = unitVector(r.direction());
	auto t = 0.5f * (unit_direction.y() + 1.0f);
	return (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
}

int main()
{
	// Image
	const float aspect_ratio = 16.0f / 9.0f;
	const int image_width = 400;
	const int image_height = static_cast<int>(image_width / aspect_ratio);
	const int max_depth = 50;
	const int samples_per_pixel = 100;

	// World
	HittableList world;


//	world.add(make_shared<Sphere>(Point(0.0f, 0.0f, -1.0f), 0.5f));
//	world.add(make_shared<Sphere>(Point(0.0f, -100.5f, -1.0f), 100.0f));

	auto material_ground = make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
//	auto material_center = make_shared<Dielectric>(1.5);
	auto material_left = make_shared<Dielectric>(1.5);
    auto material_center = make_shared<Lambertian>(Color(0.1, 0.2, 0.5));
//    auto material_left   = make_shared<Metal>(Color(0.8, 0.8, 0.8), 0.3);
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

	using namespace std;
	cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

	for (int j = image_height-1; j>=0; --j)
	{
		cerr << "\rScanlines remaining: " << j << ' ' << flush;
		for (int i = 0; i < image_width; ++i)
		{
            Color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                auto u = (i + randomFloat()) / (image_width - 1);
                auto v = (j + randomFloat()) / (image_height - 1);
                Ray r = cam.getRay(u, v);
                pixel_color += rayColor(r, world, max_depth);
            }
            writeColor(std::cout, pixel_color, samples_per_pixel);
		}
	}

	cerr << "\nDone.\n";
}