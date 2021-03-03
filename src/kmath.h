#pragma once
#include<algorithm>
#include<iostream>
using Real = float;

// some basic vector operations
template <typename T>
struct Vec2
{
    T x, y;
    Vec2(T x = 0, T y = 0) : x(x), y(y) {}
};
template <typename T>
struct Vec3
{
    T x, y, z;
    Vec3(T x = 0, T y = 0, T z = 0) : x(x), y(y), z(z) {}
};
using Vec2f = Vec2<Real>;
using Vec3i = Vec3<int>;
using Vec3f = Vec3<Real>;
Vec2f operator+(const Vec2f &v0, const Vec2f &v1) { return Vec2f{v0.x + v1.x, v0.y + v1.y}; }
Vec2f &operator+=(Vec2f &v0, const Vec2f &v1)
{
    v0.x += v1.x;
    v0.y += v1.y;
    return v0;
}
Vec2f operator-(const Vec2f &v0, const Vec2f &v1) { return Vec2f{v0.x - v1.x, v0.y - v1.y}; }
Vec2f operator*(Real s, const Vec2f &v) { return Vec2f{v.x * s, v.y * s}; }
Vec2f operator*(const Vec2f &v, Real s) { return Vec2f{v.x * s, v.y * s}; }
Vec2f operator/(const Vec2f &v, Real s) { return Vec2f{v.x / s, v.y / s}; }
Real dot(const Vec2f &v0, const Vec2f &v1) { return v0.x * v1.x + v0.y * v1.y; }
Real length(const Vec2f &v) { return sqrt(dot(v, v)); }
Vec2f normal(const Vec2f &v) { return Vec2f{-v.y, v.x}; }
Vec3f &operator+=(Vec3f &v0, const Vec3f &v1)
{
    v0.x += v1.x;
    v0.y += v1.y;
    v0.z += v1.z;
    return v0;
}
Vec3f operator+(const Vec3f &v0, const Vec3f &v1) { return Vec3f{v0.x + v1.x, v0.y + v1.y, v0.z + v1.z}; }
Vec3f operator-(const Vec3f &v0, const Vec3f &v1) { return Vec3f{v0.x - v1.x, v0.y - v1.y, v0.z - v1.z}; }
Vec3f operator*(const Vec3f &v0, const Vec3f &v1) { return Vec3f{v0.x * v1.x, v0.y * v1.y, v0.z * v1.z}; }
Vec3f operator-(const Vec3f &v) { return Vec3f{-v.x, -v.y, -v.z}; }
Vec3f operator*(const Vec3f &v, Real s) { return Vec3f{v.x * s, v.y * s, v.z * s}; }
Vec3f operator*(Real s, const Vec3f &v) { return Vec3f{v.x * s, v.y * s, v.z * s}; }
Vec3f operator/(const Vec3f &v, Real s) { return Vec3f{v.x / s, v.y / s, v.z / s}; }
Real dot(const Vec3f &v0, const Vec3f &v1) { return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z; }
Real length(const Vec3f &v) { return sqrt(dot(v, v)); }
Vec3f normalize(const Vec3f &v){auto temp=1.0/length(v);return temp*v;}
template <typename T>
T clamp(T v, T l, T u)
{
    if (v < l)
        return l;
    else if (v > u)
        return u;
    return v;
}