#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "NoiseKernel.cuh"

__device__ float clamp01(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

// this is for smoother interpolation between noise values
__device__ float fade(float t) {
    return t * t * (3.0f - 2.0f * t);
}

__device__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// Deterministic hash: maps integer 3D grid coordinates to a pseudo-random value in [0, 1).
__device__ float hash3i(int x, int y, int z) {
    unsigned int h = (unsigned int)x * 374761393u
        + (unsigned int)y * 668265263u
        + (unsigned int)z * 2147483647u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (float)(h & 0x00ffffff) / (float)0x01000000;
}

// this is where the value noise is generated.
__device__ float valueNoise(float3 p) {

    // int coordinate of the lower-left-near corner of the cell containing p
    int x0 = (int)floorf(p.x);
    int y0 = (int)floorf(p.y);
    int z0 = (int)floorf(p.z);

    // fractional position of p in the cell, each in [0, 1)
    float fx = p.x - (float)x0;
    float fy = p.y - (float)y0;
    float fz = p.z - (float)z0;

    // smooth the trasitions between cells
    float tx = fade(fx);
    float ty = fade(fy);
    float tz = fade(fz);

    //hash the 8 corners of the surrouding grid cell.
    float c000 = hash3i(x0,     y0,     z0);
    float c100 = hash3i(x0 + 1, y0,     z0);
    float c010 = hash3i(x0,     y0 + 1, z0);
    float c110 = hash3i(x0 + 1, y0 + 1, z0);
    float c001 = hash3i(x0,     y0,     z0 + 1);
    float c101 = hash3i(x0 + 1, y0,     z0 + 1);
    float c011 = hash3i(x0,     y0 + 1, z0 + 1);
    float c111 = hash3i(x0 + 1, y0 + 1, z0 + 1);

    // trilinear interpolation of the 8 corner values
    float x00 = lerp(c000, c100, tx);
    float x10 = lerp(c010, c110, tx);
    float x01 = lerp(c001, c101, tx);
    float x11 = lerp(c011, c111, tx);

    // interpolate the x results along y
    float y0v = lerp(x00, x10, ty);
    float y1v = lerp(x01, x11, ty);

    // interpolate the y along z to get fianl noise value at p
    return lerp(y0v, y1v, tz);
}

// fbm = fractal Brownian motion. It layers multiple frequencies of noise together.
// Lower frequencies contribute to the large overall shape, while higher frequencies add detail.
__device__ float fbm(float3 p) {
    float amplitude = 0.5f;
    float frequency = 1.0f;
    float total = 0.0f;

    for (int octave = 0; octave < 5; ++octave) {
        total += amplitude * valueNoise(make_float3(p.x * frequency, p.y * frequency, p.z * frequency));
        frequency *= 2.0f;
        amplitude *= 0.5f;
    }

    return total;
}

// the actual generate GPU kernel for each voxel.
//It do three main things: 1) computes the 3D position for each voxel, 2) use FBM noise to create a natural looking density, 3)  writes that density into volume[idx].
__global__ void generateNoiseKernel(float* volume, int width, int height, int depth, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth)
        return;

    float nx = (float)x / (float)(width - 1);
    float ny = (float)y / (float)(height - 1);
    float nz = (float)z / (float)(depth - 1);

    float3 p = make_float3(nx * 4.0f, ny * 4.0f, nz * 4.0f + time);
    float noise = fbm(p);

    float cx = nx * 2.0f - 1.0f;
    float cy = ny * 2.0f - 1.0f;
    float cz = nz * 2.0f - 1.0f;
    float radius = sqrtf(cx * cx + cy * cy + cz * cz);
    float falloff = clamp01(1.15f - radius);

    float density = clamp01((noise - 0.20f) * 2.2f) * falloff;

    // the index it linearly mapped from 3D to 1D, and we assign the density value to the volume at that index.
    int idx = x + y * width + z * width * height;
    volume[idx] = density;
}

// Host function to launch the noise kernel, the CUDA kernel would launch to fill a 3D density volume.
void launchNoiseKernel(float* volume, int width, int height, int depth, float time) {
    dim3 block(8, 8, 8);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        (depth + block.z - 1) / block.z
    );

    generateNoiseKernel<<<grid, block>>>(volume, width, height, depth, time);
}
