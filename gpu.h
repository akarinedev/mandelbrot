#pragma once

#include "frameinfo.h"

namespace gpu
{
	__host__ void mandelbrot(long*, frameinfo);
	__global__ void pixelcalc(long*, frameinfo);
};
