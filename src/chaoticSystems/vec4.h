#pragma once
template <typename T = double>
struct vec4_t {
    T x; T y; T z; T w;
    vec4_t operator+(const vec4_t& r) const { return {x + r.x, y + r.y, z + r.z , w + r.w}; }
    vec4_t operator*(T s) const             { return {x * s, y * s, z * s , w * s}; }
};