#include "gpu.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "frameinfo.h"

__global__
void gpu::pixelcalc(unsigned long* out, frameinfo frame)
{
	int threadnum = blockIdx.x * blockDim.x + threadIdx.x;

	int x = threadnum % frame.resx;
	int y = threadnum / frame.resx;

	if(x > frame.resx || y > frame.resy)
	{
		return;
	}

	double cx = frame.centerx - frame.scalex + frame.deltax * x;
	double cy = frame.centery - frame.scaley + frame.deltay * y;

	double zx = 0.0;
	double zy = 0.0;

	double zx2 = 0.0;
	double zy2 = 0.0;

	double zxt, zyt;

	for(long i = 0; i < frame.iters; i++)
	{
		zxt = zx2 - zy2 + cx;
		zyt = 2 * zx * zy + cy;

		zx = zxt;
		zy = zyt;

		zx2 = zx * zx;
		zy2 = zy * zy;

		#ifdef abs
		if(zx2 + zy2 >= 4)
		#else
		if(sqrt(zx2 + zy2) >= 2)
		#endif
		{
			out[threadnum] = i;
			return;
		}
	}

	out[threadnum] = frame.iters;
	return;
}

__host__
void gpu::mandelbrot(unsigned long* out, frameinfo frame)
{
	int pixels = frame.resx * frame.resy;
	int threads_per_block = 128;
	int blocks = (int) ceil((float) pixels / threads_per_block);

	if(frame.resx > frame.resy)
	{
		frame.scaley = frame.scale;
		frame.scalex = frame.scale * frame.resx / frame.resy;
	}
	else
	{
		frame.scalex = frame.scale;
		frame.scaley = frame.scale * frame.resy / frame.resx;
	}

	frame.deltax = frame.scalex / frame.resx * 2;
	frame.deltay = frame.scaley / frame.resy * 2;

	pixelcalc<<<blocks, threads_per_block>>>(out, frame);

	cudaDeviceSynchronize();
}
