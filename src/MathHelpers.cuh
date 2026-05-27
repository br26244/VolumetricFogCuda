#pragma once
#include <cuda_runtime.h>
#include <math.h>

//i didnt want to go throught the hastle of installing or figuring out
// helper_math.h so i just coppied the functions i needed

inline __host__ __device__ float3 normalize(float3 v) {
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    return make_float3(v.x/len, v.y/len, v.z/len);
}

inline __host__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __host__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __host__ __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x*b, a.y*b, a.z*b);
}

inline __host__ __device__ float3 operator*(float a, float3 b) {
    return make_float3(a*b.x, a*b.y, a*b.z);
}

inline __host__ __device__ float dot(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __host__ __device__ float length(float3 v) {
    return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

inline __host__ __device__ float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

inline __host__ __device__ float clampf(float val, float lo, float hi) {
    return fminf(fmaxf(val, lo), hi);
}