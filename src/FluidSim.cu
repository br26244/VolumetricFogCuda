#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "FluidSim.cuh"
#include "TexturePipeline.cuh"
#include "MathHelpers.cuh"
//array to hold the new density values after advection step
static float* d_densityNew = nullptr;
static int volW, volH, volD; // volume dimensions

void initFluidSim(int width, int height, int depth) {
    volW = width;
    volH = height;
    volD = depth;
    cudaMalloc(&d_densityNew, sizeof(float) * width * height * depth);
}

void freeFluidSim() {
    if (d_densityNew) cudaFree(d_densityNew);
}
//we start in advection step, we will move the density according to the velocity field
//velocity of which the particles move
__device__ float3 velocityField(float3 p, float time) {
    float x = p.x * 2.0f - 1.0f;
    float y = p.y * 2.0f - 1.0f;
    float z = p.z * 2.0f - 1.0f;

    // away from center
    float dist = sqrtf(x*x + y*y + z*z) + 1e-5f;
    float3 radial = make_float3(x / dist, y / dist, z / dist);

    // Expansion strength
    float expansion = 0.5f * (1.0f - clampf(dist, 0.0f, 1.0f));

    // swirling motion
    float vx = radial.x * expansion + sinf(y * 3.0f + time) * 0.1f;
    float vy = radial.y * expansion + cosf(x * 3.0f + time * 0.7f) * 0.1f;
    float vz = radial.z * expansion + sinf(x * 3.0f + z * 2.0f + time * 0.5f) * 0.1f;

    return make_float3(vx, vy, vz);
}
//langrangian advection 
//density vs velocity field, we want to move the density according to the velocity field
__global__ void advectKernel(cudaTextureObject_t densityTex, float* densityOut, int width, int height, int depth, float dt, float time)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= width || y >= height || z >= depth) return;

    // Voxel center in texture coords 
    float3 p = make_float3(
        (x + 0.5f) / (float)width,
        (y + 0.5f) / (float)height,
        (z + 0.5f) / (float)depth
    );

    float3 vel = velocityField(p, time); //velocity , time and position

   //density current poistion
    float3 samplePos = make_float3(
        p.x - vel.x * dt * 0.5f,
        p.y - vel.y * dt * 0.5f,
        p.z - vel.z * dt * 0.5f
    );

    samplePos.x = clampf(samplePos.x, 0.0f, 1.0f); 
    samplePos.y = clampf(samplePos.y, 0.0f, 1.0f);
    samplePos.z = clampf(samplePos.z, 0.0f, 1.0f);

    // Sample the density at the backtraced position
    float advected = tex3D<float>(densityTex, samplePos.x, samplePos.y, samplePos.z);
    advected *= 0.975; //decay

    //motion turbulence
    float3 wp = make_float3(p.x * 8.0f, p.y * 8.0f, p.z * 8.0f + time * 0.3f);
    //swirls
    float turbulence = __sinf(wp.x*1.7f + wp.y*2.3f) * sinf(wp.y*1.9f + wp.z*1.3f) * sinf(wp.z*2.1f + wp.x*1.1f);
    turbulence = (turbulence + 1.0f) * 0.5f;  
    
    advected += turbulence * 0.02f * advected; //
    advected = clampf(advected, 0.0f, 1.0f);

    // Emitter 
    //its curr at the top
    float cx = p.x - 0.5f;
    float cz = p.z - 0.5f;
    float distXZ = sqrtf(cx * cx + cz * cz);
    float emitter = 0.0f;
    if (p.y < 0.12f) {
        emitter = expf(-distXZ * distXZ * 60.0f) * 0.2f;
    }

    float result = fmaxf(advected, emitter);
    result = clampf(result, 0.0f, 1.0f);

    int idx = x + y * width + z * width * height; //3D to 1D mapping
    densityOut[idx] = result;
}

void stepFluidSim(float time, float dt) {
    dim3 block(8, 8, 8);
    dim3 grid(
        (volW + 7) / 8,
        (volH + 7) / 8,
        (volD + 7) / 8
    );

    advectKernel<<<grid, block>>>(volumeTex, d_densityNew, volW, volH, volD, dt, time);
    cudaDeviceSynchronize();
    //bounce the texture with new valuess
    updateVolumeTexture(d_densityNew);
}