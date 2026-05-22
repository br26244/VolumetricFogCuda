#include <cuda_runtime.h>
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void launchRenderKernel(uchar4* buffer, int width, int height);

int main() {
    const int width  = 800;
    const int height = 600;
    const int pixels = width * height;

    // Allocate output buffer on GPU
    uchar4* d_buffer;
    cudaMalloc(&d_buffer, pixels * sizeof(uchar4));

    // Run the kernel
    launchRenderKernel(d_buffer, width, height);
    cudaDeviceSynchronize();

    // Copy result back to CPU
    uchar4* h_buffer = new uchar4[pixels];
    cudaMemcpy(h_buffer, d_buffer, pixels * sizeof(uchar4), cudaMemcpyDeviceToHost);

    // Save to PNG
    stbi_write_png("output.png", width, height, 4, h_buffer, width * 4);
    std::cout << "Saved output.png" << std::endl;

    // Cleanup
    delete[] h_buffer;
    cudaFree(d_buffer);
    return 0;
}