
#ifndef  RAY_MARCHER_CUH
#define  RAY_MARCHER_CUH

#include <cuda_runtime.h>

//uchar4 is for rgb and a 
void launchRenderKernel(uchar4* buffer, int width, int height, float time);

#endif