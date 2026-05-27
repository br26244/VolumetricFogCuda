#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "TexturePipeline.cuh"
#include <stdio.h>
#include <iostream>

#define VOLUME_WIDTH 128
#define VOLUME_HEIGHT 128
#define VOLUME_DEPTH 128

cudaTextureObject_t volumeTex = 0;
cudaArray_t d_volumeArray =  nullptr;

void initVolumeTexture(){
    // each 3d/2d pixel is a float
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    //3d array
    cudaExtent volumeSize = make_cudaExtent(VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH);

    //error check
    cudaError_t err =  cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize);
    if (err != cudaSuccess) {
        std::cout << "Failed to allocate 3D array: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    //density
    int totalVoxels = VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH;
    float* h_volume = new float[totalVoxels];
    for(int i = 0; i < totalVoxels; ++i){
        h_volume[i] = 0.5f; //can be changed
    }

    //data to array. sending h_volume to d_volumeArray
    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr = make_cudaPitchedPtr(h_volume, VOLUME_WIDTH * sizeof(float), VOLUME_WIDTH, VOLUME_HEIGHT);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyHostToDevice;

    //error check
    err = cudaMemcpy3D(&copyParams);
    if (err != cudaSuccess) {
        std::cout << "Failed to copy data to 3D array: " << cudaGetErrorString(err) << std::endl;
        delete[] h_volume;
        return;
    }
    //we were able to pass the data to d_volumeArray so we can delete it
    delete[] h_volume;

    //texture objext
    cudaResourceDesc resDesc = {};              //empty resource descriptor
    resDesc.resType = cudaResourceTypeArray;    //descriptor is an array
    resDesc.res.array.array = d_volumeArray;    //we have d_volumeArray be assigned to the type array

    cudaTextureDesc texDesc = {};              //empty texture 
    texDesc.addressMode[0] = cudaAddressModeClamp; //cant go out of range (x,y,z )
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear; //lin interpolation
    texDesc.readMode = cudaReadModeElementType; //float
    texDesc.normalizedCoords = 1; //we normalize

    //error check
    err = cudaCreateTextureObject(&volumeTex, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        std::cout << "Failed to create texture object: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "Volume texture initialized successfully with " << VOLUME_WIDTH 
    << " x " << VOLUME_HEIGHT << " x " << VOLUME_DEPTH << " voxels" << std::endl;

}

void freeVolumeTexture(){
    if(volumeTex) {
        cudaDestroyTextureObject(volumeTex);
    }
    if(d_volumeArray) {
        cudaFreeArray(d_volumeArray);
    }
}


