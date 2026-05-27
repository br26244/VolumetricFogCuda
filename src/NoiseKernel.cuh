#ifndef NOISE_KERNEL_CUH
#define NOISE_KERNEL_CUH
// NoiseKernel.cuh， this is the header file for the noise kernel.
void launchNoiseKernel(float* volume, int width, int height, int depth, float time);

#endif
