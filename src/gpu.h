#pragma once

#include "frameinfo.h"

namespace gpu
{
	__host__ void mandelbrot(unsigned long*, frameinfo);
	__global__ void pixelcalc(unsigned long*, frameinfo);
	__host__ long itercount(frameinfo);
};
