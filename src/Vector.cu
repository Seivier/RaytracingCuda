//
// Created by vigb9 on 19/07/2023.
//
#include "Raytracing/Utils.cuh"
#include "Raytracing/Vector.cuh"

#include <glm/gtx/norm.hpp>



Vector Vector::operator-() const
{
	return Vector(-e[0], -e[1], -e[2]);
}

float Vector::operator[](int i) const
{
	return e[i];
}

float& Vector::operator[](int i)
{
	return e[i];
}

Vector& Vector::operator+=(const Vector& v)
{
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];

	return *this;
}

Vector& Vector::operator*=(const float t)
{
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;

	return *this;
}

Vector& Vector::operator/=(const float t)
{
	return *this *= 1 / t;
}

float Vector::length() const
{
	return sqrtf(squaredLength());
}

float Vector::squaredLength() const
{
	return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
}


bool Vector::nearZero() const
{
	const auto s = 1e-8;
	return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
}

std::ostream& operator<<(std::ostream& out, const Vector& v)
{
	return out << v.x()	 << ' ' << v.y() << ' ' << v.z();
}

Vector operator+(const Vector& u, const Vector& v)
{
	return Vector(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

Vector operator-(const Vector& u, const Vector& v)
{
	return Vector(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

Vector operator*(const Vector& u, const Vector& v)
{
	return Vector(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

Vector operator*(float t, const Vector& v)
{
	return Vector(t * v.x(), t * v.y(), t * v.z());
}

Vector operator*(const Vector& v, float t)
{
	return t * v;
}

Vector operator/(Vector v, float t)
{
	return (1 / t) * v;
}

float dot(const Vector& u, const Vector& v)
{
	return u.x() * v.x()
		+ u.y() * v.y()
		+ u.z() * v.z();
}

Vector cross(const Vector& u, const Vector& v)
{
	return Vector(u.y() * v.z() - u.z() * v.y(),
		u.z() * v.x() - u.x() * v.z(),
		u.x() * v.y() - u.y() * v.x());
}

Vector unitVector(Vector v)
{
	return v / v.length();
}


Vector randomInUnitSphere()
{
	while (true)
	{
		auto p = random(-1.f, 1.f);
		if (p.squaredLength() >= 1) continue;
		return p;
	}
}

Vector randomUnitVector()
{
	return unitVector(randomInUnitSphere());
}

Vector randomInHemisphere(const Vector& normal)
{
	Vector in_unit_sphere = randomInUnitSphere();
	if (dot(in_unit_sphere, normal) > 0.0f) // In the same hemisphere as the normal
		return in_unit_sphere;
	else
		return -in_unit_sphere;
}

Vector reflect(const Vector& v, const Vector& n) {
	return v - 2*dot(v,n)*n;
}

Vector refract(const Vector& uv, const Vector& n, float etai_over_etat)
{
	auto cos_theta = fmin(dot(-uv, n), 1.f);
	Vector r_out_parallel = etai_over_etat * (uv + cos_theta * n);
	Vector r_out_perp = -sqrtf(1.0f - r_out_parallel.squaredLength()) * n;
	return r_out_parallel + r_out_perp;
}

Vector randomInUnitDisk() {
    while (true) {
        auto p = Vector(randomFloat(-1,1), randomFloat(-1,1), 0);
        if (p.squaredLength() >= 1) continue;
        return p;
    }
}