//
// Created by vigb9 on 20/07/2023.
//

#ifndef _VECTOR_CUH_
#define _VECTOR_CUH_

#define CUDA_CALLABLE __host__ __device__
#define CUDA_ONLY __device__

#include <cmath>
#include <cstdlib>
#include <iostream>

class vector
{
 public:
	CUDA_CALLABLE vector() {}
    CUDA_CALLABLE vector(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
    CUDA_CALLABLE inline float x() const { return e[0]; }
    CUDA_CALLABLE inline float y() const { return e[1]; }
    CUDA_CALLABLE inline float z() const { return e[2]; }
    CUDA_CALLABLE inline float r() const { return e[0]; }
    CUDA_CALLABLE inline float g() const { return e[1]; }
    CUDA_CALLABLE inline float b() const { return e[2]; }

    CUDA_CALLABLE inline vector operator-() const { return vector(-e[0], -e[1], -e[2]); }
    CUDA_CALLABLE inline float operator[](int i) const { return e[i]; }
    CUDA_CALLABLE inline float& operator[](int i) { return e[i]; };

    CUDA_CALLABLE vector& operator+=(const vector &v2);
    CUDA_CALLABLE vector& operator-=(const vector &v2);
    CUDA_CALLABLE vector& operator*=(const vector &v2);
    CUDA_CALLABLE vector& operator/=(const vector &v2);
    CUDA_CALLABLE vector& operator*=(float t);
    CUDA_CALLABLE vector& operator/=(float t);

    CUDA_CALLABLE float length() const;
    CUDA_CALLABLE float squaredLength() const;
    CUDA_CALLABLE void makeUnitVector();

 private:
	float e[3];
};

std::iostream& operator<<(std::iostream &out, const vector &v);
std::istream& operator>>(std::istream &in, vector &v);

CUDA_CALLABLE vector operator+(const vector &v1, const vector &v2);
CUDA_CALLABLE vector operator-(const vector &v1, const vector &v2);
CUDA_CALLABLE vector operator*(const vector &v1, const vector &v2);
CUDA_CALLABLE vector operator/(const vector &v1, const vector &v2);
CUDA_CALLABLE vector operator*(float t, const vector &v);
CUDA_CALLABLE vector operator/(vector v, float t);
CUDA_CALLABLE vector operator*(const vector &v, float t);
CUDA_CALLABLE float dot(const vector &v1, const vector &v2);
CUDA_CALLABLE vector cross(const vector &v1, const vector &v2);
CUDA_CALLABLE vector unitVector(vector v);

using point = vector;
using color = vector;

#endif //_VECTOR_CUH_
