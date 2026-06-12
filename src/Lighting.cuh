#pragma once
#include <cuda_runtime.h>
#include "MathHelpers.cuh"

struct VolumeLight {
    float3 direction;
    float3 color;
    float intensity;
};

// Multi-colored directional lights. The fog density stays neutral; these colors
// only affect the light scattered by the volume.
#define NUM_VOLUME_LIGHTS 3
#define SUN_INTENSITY 2.5f
// Phase function asymmetry 
#define PHASE_G 0.1f

// Shadow ray 
#define SHADOW_STEPS     8
#define SHADOW_STEP_SIZE 0.03f

__device__ inline VolumeLight getVolumeLight(int index, float time) {
    VolumeLight light;

    if (index == 0) {
        // Warm key light: orange/yellow sunlight from upper-left.
        light.direction = normalize(make_float3(-0.5f, 0.8f, -0.3f));
        light.color = make_float3(1.0f, 0.82f, 0.55f);
        light.intensity = 2.2f;
    } else if (index == 1) {
        // Cool side light: blue light from the opposite side, slightly animated.
        light.direction = normalize(make_float3(0.75f, 0.25f, -0.55f));
        light.color = make_float3(0.35f, 0.55f, 1.0f);
        light.intensity = 1.35f + 0.25f * sinf(time * 0.7f);
    } else {
        // Soft fill light: green/cyan light from below/front, slightly animated.
        light.direction = normalize(make_float3(-0.2f, -0.35f, -0.9f));
        light.color = make_float3(0.45f, 1.0f, 0.78f);
        light.intensity = 0.85f + 0.15f * cosf(time * 0.5f);
    }

    return light;
}

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
