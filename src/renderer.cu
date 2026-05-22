#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fillKernel(uchar4* buffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) 
        return;

    int idx = y * width + x;
    buffer[idx] = make_uchar4( (unsigned char)(255.0f * x / width), (unsigned char)(255.0f * y / height),
        128, 255);
}

void launchRenderKernel(uchar4* buffer, int width, int height) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    fillKernel<<<grid, block>>>(buffer, width, height);
}