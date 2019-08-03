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

	double cx = frame.winl + frame.deltax * x;
	double cy = frame.wint + frame.deltay * y;

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

		if(zx2 + zy2 >= 4)
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

	std::cout << "Pixels: " << pixels << std::endl;
	std::cout << "Threads per block: " << threads_per_block << std::endl;
	std::cout << "Blocks: " << blocks << std::endl;

	frame.deltax = (frame.winr - frame.winl) / (double) (frame.resx - 1);
	frame.deltay = (frame.winb - frame.wint) / (double) (frame.resy - 1);

	std::cout << "Starting GPU Compute" << std::endl;

	pixelcalc<<<blocks, threads_per_block>>>(out, frame);

	cudaDeviceSynchronize();

	std::cout << "Finished GPU Compute" << std::endl;
}
