//
// Created by vigb9 on 20/07/2023.
//

#include "vector.cuh"
CUDA_CALLABLE vector& vector::operator+=(const vector& v2)
{
	e[0] += v2[0];
	e[1] += v2[1];
	e[2] += v2[2];
	return *this;
}
CUDA_CALLABLE vector& vector::operator-=(const vector& v2)
{
	e[0] -= v2[0];
	e[1] -= v2[1];
	e[2] -= v2[2];
	return *this;
}
CUDA_CALLABLE vector& vector::operator*=(const vector& v2)
{
	e[0] *= v2[0];
	e[1] *= v2[1];
	e[2] *= v2[2];
	return *this;
}
CUDA_CALLABLE vector& vector::operator/=(const vector& v2)
{
	e[0] /= v2[0];
	e[1] /= v2[1];
	e[2] /= v2[2];
	return *this;
}
CUDA_CALLABLE vector& vector::operator*=(float t)
{
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}
CUDA_CALLABLE vector& vector::operator/=(float t)
{
	float k = 1.0f / t;
	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
	return *this;
}
CUDA_CALLABLE float vector::length() const
{
	return sqrtf(squaredLength());
}
CUDA_CALLABLE float vector::squaredLength() const
{
	return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
}
CUDA_CALLABLE void vector::makeUnitVector()
{
	float k = 1.f / length();
	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
}
std::iostream& operator<<(std::iostream& out, const vector& v)
{
	out << v[0] << " " << v[1] << " " << v[2];
	return out;
}
std::istream& operator>>(std::istream& in, vector& v)
{
	in >> v[0] >> v[1] >> v[2];
	return in;
}

CUDA_CALLABLE vector operator+(const vector& v1, const vector& v2)
{
	return vector(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);
}
CUDA_CALLABLE vector operator-(const vector& v1, const vector& v2)
{
	return vector(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]);
}
CUDA_CALLABLE vector operator*(const vector& v1, const vector& v2)
{
	return vector(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]);
}
CUDA_CALLABLE vector operator/(const vector& v1, const vector& v2)
{
	return vector(v1[0] / v2[0], v1[1] / v2[1], v1[2] / v2[2]);
}
CUDA_CALLABLE vector operator*(float t, const vector& v)
{
	return vector(t * v[0], t * v[1], t * v[2]);
}
CUDA_CALLABLE vector operator/(vector v, float t)
{
	return vector(v[0] / t, v[1] / t, v[2] / t);
}
CUDA_CALLABLE vector operator*(const vector& v, float t)
{
	return vector(t * v[0], t * v[1], t * v[2]);
}
CUDA_CALLABLE float dot(const vector& v1, const vector& v2)
{
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}
CUDA_CALLABLE vector cross(const vector& v1, const vector& v2)
{
	return vector((v1[1] * v2[2] - v1[2] * v2[1]),
		(-(v1[0] * v2[2] - v1[2] * v2[0])),
		(v1[0] * v2[1] - v1[1] * v2[0]));
}
CUDA_CALLABLE vector unitVector(vector v)
{
	return v / v.length();
}