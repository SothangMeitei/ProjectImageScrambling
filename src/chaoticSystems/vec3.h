#pragma once
template <typename T = double>
struct vec3_t {
    T x; T y; T z;
    vec3_t operator+(const vec3_t& r) const { return {x + r.x, y + r.y, z + r.z}; }
    vec3_t operator*(T s) const             { return {x * s, y * s, z * s}; }
};