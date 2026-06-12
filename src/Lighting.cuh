#pragma once
#include <cuda_runtime.h>
#include "MathHelpers.cuh"

// Sun color and intensity
#define SUN_COLOR_R   1.0f
#define SUN_COLOR_G   0.95f
#define SUN_COLOR_B   0.95f
#define SUN_INTENSITY 2.5f

// Phase function asymmetry 
#define PHASE_G 0.1f

// Shadow ray settings
#define SHADOW_STEPS     8
#define SHADOW_STEP_SIZE 0.03f

// Henyey Greenstein phase function for scattering
__device__ inline float phaseHG(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cosTheta;
    return (1.0f - g2) / (4.0f * 3.14159265f * denom * sqrtf(denom));
}

// Shadow marching to estimate light visibility from a point in the volume
__device__ inline float shadowMarch(
    float3 pos, float3 lightDir,
    cudaTextureObject_t volTex,
    float extinction, float time)
{
    float shadowT = 1.0f; // Start fully lit
    float3 samplePos = pos;

    for (int i = 0; i < SHADOW_STEPS; i++) {
        samplePos.x += lightDir.x * SHADOW_STEP_SIZE; // Move along the light direction
        samplePos.y += lightDir.y * SHADOW_STEP_SIZE;
        samplePos.z += lightDir.z * SHADOW_STEP_SIZE;

        float tx = (samplePos.x + 1.0f) * 0.5f; 
        float ty = (samplePos.y + 1.0f) * 0.5f;
        float tz = (samplePos.z + 1.0f) * 0.5f;

        // Outside volume 
        if (tx < 0.0f || tx > 1.0f ||
            ty < 0.0f || ty > 1.0f ||
            tz < 0.0f || tz > 1.0f) break;

        float density = tex3D<float>(volTex, tx, ty, tz); // Sample density along the shadow ray
        shadowT *= expf(-density * extinction * SHADOW_STEP_SIZE); // Beer Lambert attenuation

        if (shadowT < 0.01f) return 0.0f; 
    }

    return shadowT;
}