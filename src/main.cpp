#include <cuda_runtime.h>
#include <iostream>

//calls for png writing
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "TexturePipeline.cuh"
#include "RayMarcher.cuh"

void launchRenderKernel(uchar4* buffer, int width, int height);

int main() {
    const int width  = 800;
    const int height = 600;
    const int pixels = width * height;

    initVolumeTexture();

    uchar4* d_buffer;
    cudaMalloc(&d_buffer, pixels * sizeof(uchar4));

    launchRenderKernel(d_buffer, width, height);
    cudaDeviceSynchronize();

    //error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    uchar4* h_buffer = new uchar4[pixels];
    cudaMemcpy(h_buffer, d_buffer, pixels * sizeof(uchar4), cudaMemcpyDeviceToHost);
    stbi_write_png("output.png", width, height, 4, h_buffer, width * 4);
    std::cout << "Saved output.png" << std::endl;

    delete[] h_buffer;
    cudaFree(d_buffer);
    freeVolumeTexture();
    return 0;
}