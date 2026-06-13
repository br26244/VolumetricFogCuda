#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "TexturePipeline.cuh"
#include "RayMarcher.cuh"
#include "NoiseKernel.cuh"
#include "FluidSim.cuh"

int main() {
    const int width  = 800;
    const int height = 600;
    const int pixels = width * height;

    // Init texture pipeline
    initVolumeTexture();

    // Initial density fill via noise kernel
    float* d_initialDensity;
    cudaMalloc(&d_initialDensity, sizeof(float) * VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH);
    launchNoiseKernel(d_initialDensity, VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH, 0.0f);
    cudaDeviceSynchronize();
    updateVolumeTexture(d_initialDensity);

    // Init fluid sim
    initFluidSim(VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH);

    // Output buffer
    uchar4* d_buffer;
    cudaMalloc(&d_buffer, pixels * sizeof(uchar4));
    uchar4* h_buffer = new uchar4[pixels];

    const int   NUM_FRAMES = 240;
    const float dt = 0.05f;
    float time = 0.0f;

    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        // Advect the density volume
        stepFluidSim(time, dt);

        float cameraAngle = (2.0f * 3.14159265f * frame) / NUM_FRAMES;
        // Render current state
        launchRenderKernel(d_buffer, width, height, time, cameraAngle);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel error: " << cudaGetErrorString(err) << std::endl;
            break;
        }

        cudaMemcpy(h_buffer, d_buffer, pixels * sizeof(uchar4), cudaMemcpyDeviceToHost);

        char filename[64];
        snprintf(filename, sizeof(filename), "../png/frame_%04d.png", frame);
        stbi_write_png(filename, width, height, 4, h_buffer, width * 4);

        printf("Rendered frame %d/%d\n", frame, NUM_FRAMES);

        time += dt;
    }

    delete[] h_buffer;
    cudaFree(d_buffer);
    cudaFree(d_initialDensity);
    freeFluidSim();
    freeVolumeTexture();

    return 0;
}