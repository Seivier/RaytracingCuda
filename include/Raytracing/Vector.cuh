#ifndef _VECTOR_CUH_
#define _VECTOR_CUH_

#include <cmath>
#include <iostream>


class Vector
{
 public:
	Vector() : e{0, 0, 0} {}
	Vector(float e0, float e1, float e2) : e{e0, e1, e2} {}

	float x() const { return e[0]; }
	float y() const { return e[1]; }
	float z() const { return e[2]; }

	Vector operator-() const;
	float operator[](int i) const;
	float& operator[](int i);

	Vector& operator+=(const Vector& v);

	Vector& operator*=(float t);

	Vector& operator/=(float t);

	float length() const;

	float squaredLength() const;

	bool nearZero() const;

 private:
	float e[3];
};


// Type aliases for Vector
using Point = Vector;
using Color = Vector;

// Vector Utility Functions
std::ostream& operator<<(std::ostream& out, const Vector& v);
Vector operator+(const Vector& u, const Vector& v);
Vector operator-(const Vector& u, const Vector& v);
Vector operator*(const Vector& u, const Vector& v);
Vector operator*(float t, const Vector& v);
Vector operator*(const Vector& v, float t);
Vector operator/(Vector v, float t);
float dot(const Vector& u, const Vector& v);
Vector cross(const Vector& u, const Vector& v);
Vector unitVector(Vector v);


void writeColor(std::ostream& out, Color pixel_color, int samples_per_pixel);

float randomFloat();
float randomFloat(float min, float max);

inline Vector random()
{
	return { randomFloat(), randomFloat(), randomFloat()};
}

inline Vector random(float min, float max)
{
	return { randomFloat(min, max), randomFloat(min, max), randomFloat(min, max)};
}


// Diffusion methods
Vector randomInUnitSphere();

Vector randomUnitVector();

Vector randomInHemisphere(const Vector& normal);

Vector reflect(const Vector& v, const Vector& n);

Vector refract(const Vector& uv, const Vector& n, float etai_over_etat);

Vector randomInUnitDisk();

#endif //_VECTOR_CUH_
