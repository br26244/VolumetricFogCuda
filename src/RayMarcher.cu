#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "RayMarcher.cuh"
#include "TexturePipeline.cuh"
#include <stdio.h>
#include <iostream>
#include <math.h>

#include "MathHelpers.cuh"

__global__ void rayMarchKernel(uchar4* buffer, int width, int height, cudaTextureObject_t volTex){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //check if we are out of bounds
    if(x >= width || y >= height) return;

    //screen coordinates to [-1,1]
    float u = (x + 0.5f) / ((float)width) * 2.0f - 1.0f; 
    float v = (y + 0.5f) / ((float)height) * 2.0f - 1.0f;

    float3 rayOrigin = make_float3(0.0f, 0.0f, 2.0f);
    float3 rayDir = normalize(make_float3(u * 0.5f, v * 0.5f, -1.0f));

    const int maxSteps = 256;
    const float stepSize = 0.005f;
    const float extinction = 5.0f;

    //light passing through the volume
    float transmittance = 1.0f;
    float3 pos = rayOrigin;

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
        float density = tex3D<float>(volTex, tx, ty, tz);

        //beer lambert
        transmittance *= expf(-density * extinction * stepSize);

        if(transmittance < 0.01f) 
            break;  
    }


    //white fog
    float fog = 1.0f - transmittance;
    unsigned char val = (unsigned char)(fog * 255.0f);
    buffer[y * width + x] = make_uchar4(val, val, val, 255); //set each pixel to the rgba
}

void launchRenderKernel(uchar4* buffer, int width, int height){
    dim3 blockSize(16, 16);
    //15 offset for rounding up 
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    rayMarchKernel<<<gridSize, blockSize>>>(buffer, width, height, volumeTex);

}