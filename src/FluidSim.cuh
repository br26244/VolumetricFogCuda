#pragma once
#include <cuda_runtime.h>

void initFluidSim(int width, int height, int depth);
void freeFluidSim();
void stepFluidSim(float time, float dt);