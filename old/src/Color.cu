//
// Created by vigb9 on 19/07/2023.
//
#include "Raytracing/Color.cuh"

float clamp(float x, float min, float max);

void writeColor(std::ostream& out, Color pixel_color, int samples_per_pixel)
{
	auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Divide the color by the number of samples.
    auto scale = 1.f / samples_per_pixel;
    r = sqrt(scale * r);
	g = sqrt(scale * g);
	b = sqrt(scale * b);

    // Write the translated [0,255] value of each color component.
	out << static_cast<int>(256 * clamp(r, 0.0f, 0.999f)) << ' '
		<< static_cast<int>(256 * clamp(g, 0.0f, 0.999f)) << ' '
		<< static_cast<int>(256 * clamp(b, 0.0f, 0.999f)) << '\n';
}
