#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "RayMarcher.cuh"
#include "TexturePipeline.cuh"
#include <stdio.h>
#include <iostream>
#include <math.h>

#include "MathHelpers.cuh"
#include "Lighting.cuh"

__global__ void rayMarchKernel(uchar4* buffer, int width, int height, cudaTextureObject_t volTex, float time){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //check if we are out of bounds
    if(x >= width || y >= height) return;

    //screen coordinates to [-1,1]
    float u = (x + 0.5f) / ((float)width) * 2.0f - 1.0f; 
    float v = (y + 0.5f) / ((float)height) * 2.0f - 1.0f;

    float3 rayOrigin = make_float3(0.0f, 0.0f, 2.0f);
    float3 rayDir = normalize(make_float3(u * 0.5f, v * 0.5f, -1.0f));

    const int maxSteps = 500;
    const float stepSize = 0.008f;
    const float extinction = 5.0f;

    //light passing through the volume
    float transmittance = 1.0f;
    float3 pos = rayOrigin;

    float3 inScattered = make_float3(0.0f, 0.0f, 0.0f);

    for(int i = 0; i < maxSteps; i++){
        pos.x += rayDir.x * stepSize;
        pos.y += rayDir.y * stepSize;
        pos.z += rayDir.z * stepSize;

        float tx = (pos.x + 1.0f) * 0.5f;
        float ty = (pos.y + 1.0f) * 0.5f;
        float tz = (pos.z + 1.0f) * 0.5f;

        if(tx < 0.0f || tx > 1.0f || ty < 0.0f || ty > 1.0f || tz < 0.0f || tz > 1.0f) 
            continue;

        //sample 3d texture
        float movingZ = tz; //animate the fog by moving the sampling position
        movingZ = movingZ - floorf(movingZ); //wrap around to create a looping animation
        float density = tex3D<float>(volTex, tx, ty, movingZ); 
        if (density < 0.01f) {
        continue;
        }
        //in scattered light
        float sampleExtinction = density * extinction * stepSize;

        float3 ambientColor = make_float3(0.6f, 0.6f, 0.65f); // neutral gray, slight blue
        float3 ambientAmount = ambientColor * (sampleExtinction * transmittance * transmittance); 

        float3 scatterAmount = make_float3(0.0f, 0.0f, 0.0f);
        for (int lightIndex = 0; lightIndex < NUM_VOLUME_LIGHTS; ++lightIndex) {
            VolumeLight light = getVolumeLight(lightIndex, time);
            float cosTheta = dot(rayDir, light.direction);
            float phase = phaseHG(cosTheta, PHASE_G);
            float shadowT = 1.0f;
            if (lightIndex == 0) {
            shadowT = shadowMarch(pos, light.direction, volTex, extinction, time);
            }

            scatterAmount = scatterAmount + light.color *
                (light.intensity * shadowT * phase * sampleExtinction * transmittance);
        }

        inScattered = inScattered + scatterAmount + ambientAmount;


        //beer lambert
        transmittance *= expf(-density * extinction * stepSize);

        if(transmittance < 0.01f) 
            break;  
    }

    //background color
    float3 background = make_float3(0.0f, 0.0f, 0.0f);
    float smoothT = transmittance * transmittance * (3.0f - 2.0f * transmittance); 
    float3 finalColor = inScattered + background * smoothT;
    //white fog
    // pull the color towards white
    float gray = (finalColor.x + finalColor.y + finalColor.z) / 3.0f;
    float desaturation = 0.4; // amount to pull the color towards white

    finalColor.x = finalColor.x * (1.0f - desaturation) + gray * desaturation;
    finalColor.y = finalColor.y * (1.0f - desaturation) + gray * desaturation;
    finalColor.z = finalColor.z * (1.0f - desaturation) + gray * desaturation;
    unsigned char r = (unsigned char)(clampf(finalColor.x,0.0f,1.0f) * 255.0f);
    unsigned char g = (unsigned char)(clampf(finalColor.y,0.0f,1.0f) * 255.0f);
    unsigned char b = (unsigned char)(clampf(finalColor.z,0.0f,1.0f) * 255.0f);
    buffer[y * width + x] = make_uchar4(r, g, b, 255);
}

void launchRenderKernel(uchar4* buffer, int width, int height,  float time){
    dim3 blockSize(16, 16);
    //15 offset for rounding up 
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    rayMarchKernel<<<gridSize, blockSize>>>(buffer, width, height, volumeTex, time);

}
