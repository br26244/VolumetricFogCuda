
#ifndef  TEXTURE_PIPELINE_CUH
#define  TEXTURE_PIPELINE_CUH
#include <cuda_runtime.h>

#define VOLUME_WIDTH  128
#define VOLUME_HEIGHT 128
#define VOLUME_DEPTH  128

extern cudaTextureObject_t volumeTex;

void initVolumeTexture();
void freeVolumeTexture();
void updateVolumeTexture(float* d_density);

#endif 