#ifndef _VECTOR_CUH_
#define _VECTOR_CUH_

#include <cmath>
#include <iostream>

#include <curand_kernel.h>

#define CUDA_CALLABLE_MEMBER __host__ __device__


class Vector
{
 public:
	CUDA_CALLABLE_MEMBER Vector() : e{0, 0, 0} {}
	CUDA_CALLABLE_MEMBER Vector(float e0, float e1, float e2) : e{e0, e1, e2} {}
	// Move semantics
	CUDA_CALLABLE_MEMBER Vector(Vector&& v) noexcept
	{
		e[0] = v.e[0];
		e[1] = v.e[1];
		e[2] = v.e[2];
	}
	CUDA_CALLABLE_MEMBER Vector& operator=(Vector&& v) noexcept
	{
		e[0] = v.e[0];
		e[1] = v.e[1];
		e[2] = v.e[2];

		return *this;
	}
	// Copy semantics
	CUDA_CALLABLE_MEMBER Vector(const Vector& v)
	{
		e[0] = v.e[0];
		e[1] = v.e[1];
		e[2] = v.e[2];
	}
	CUDA_CALLABLE_MEMBER Vector& operator=(const Vector& v)
	{
		e[0] = v.e[0];
		e[1] = v.e[1];
		e[2] = v.e[2];

		return *this;
	}


	CUDA_CALLABLE_MEMBER float x() const { return e[0]; }
	CUDA_CALLABLE_MEMBER float y() const { return e[1]; }
	CUDA_CALLABLE_MEMBER float z() const { return e[2]; }

	CUDA_CALLABLE_MEMBER Vector operator-() const;
	CUDA_CALLABLE_MEMBER float operator[](int i) const;
	CUDA_CALLABLE_MEMBER float& operator[](int i);
	CUDA_CALLABLE_MEMBER Vector& operator+=(const Vector& v);
	CUDA_CALLABLE_MEMBER Vector& operator*=(float t);
	CUDA_CALLABLE_MEMBER Vector& operator/=(float t);




	CUDA_CALLABLE_MEMBER float length() const;

	CUDA_CALLABLE_MEMBER float squaredLength() const;

	CUDA_CALLABLE_MEMBER bool nearZero() const;

 private:
	float e[3];
};


// Type aliases for Vector
using Point = Vector;
using Color = Vector;

// Vector Utility Functions
std::ostream& operator<<(std::ostream& out, const Vector& v);
CUDA_CALLABLE_MEMBER Vector operator+(const Vector& u, const Vector& v);
CUDA_CALLABLE_MEMBER Vector operator-(const Vector& u, const Vector& v);
CUDA_CALLABLE_MEMBER Vector operator*(const Vector& u, const Vector& v);
CUDA_CALLABLE_MEMBER Vector operator*(float t, const Vector& v);
CUDA_CALLABLE_MEMBER Vector operator*(const Vector& v, float t);
CUDA_CALLABLE_MEMBER Vector operator/(Vector v, float t);
CUDA_CALLABLE_MEMBER float dot(const Vector& u, const Vector& v);
CUDA_CALLABLE_MEMBER Vector cross(const Vector& u, const Vector& v);
CUDA_CALLABLE_MEMBER Vector unitVector(Vector v);


void writeColor(std::ostream& out, Color pixel_color, int samples_per_pixel);

float randomFloat();
float randomFloat(float min, float max);

__device__ float randomFloat(curandState* state);
__device__ float randomFloat(float min, float max, curandState* state);

inline Vector random()
{
	return { randomFloat(), randomFloat(), randomFloat()};
}

__device__ Vector random(curandState* state)
{
	return { randomFloat(state), randomFloat(state), randomFloat(state)};
}

inline Vector random(float min, float max)
{
	return { randomFloat(min, max), randomFloat(min, max), randomFloat(min, max)};
}

__device__ Vector random(float min, float max, curandState* state)
{
	return { randomFloat(min, max, state), randomFloat(min, max, state), randomFloat(min, max, state)};
}


// Diffusion methods
Vector randomInUnitSphere();
Vector randomUnitVector();
Vector randomInHemisphere(const Vector& normal);
Vector randomInUnitDisk();

__device__ Vector randomInUnitSphere(curandState* state);
__device__ Vector randomUnitVector(curandState* state);
__device__ Vector randomInHemisphere(const Vector& normal, curandState* state);
__device__ Vector randomInUnitDisk(curandState* state);


CUDA_CALLABLE_MEMBER Vector reflect(const Vector& v, const Vector& n);

CUDA_CALLABLE_MEMBER Vector refract(const Vector& uv, const Vector& n, float etai_over_etat);


#endif //_VECTOR_CUH_
