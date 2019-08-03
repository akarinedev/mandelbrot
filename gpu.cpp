#include "gpu.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "frameinfo.h"

__global__
void gpu::pixelcalc(long* out, frameinfo frame)
{
	int threadnum = blockIdx.x * blockDim.x + threadIdx.x;

	int x = threadnum % frame.resx;
	int y = threadnum / frame.resx;

	if(x > frame.resx || y > frame.resy)
	{
		return;
	}

	double cx = frame.winl + frame.deltax * x;
	double cy = frame.winb + frame.deltay * y;

	double zx = 0.0;
	double zy = 0.0;

	double zx2 = 0.0;
	double zy2 = 0.0;

	double zxt, zyt;

	//out[threadnum] = cx;
	//return;

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
void gpu::mandelbrot(long* out2, frameinfo frame)
{
	int PIXELS = frame.resx * frame.resy;
	int THREADS_PER_BLOCK = 128;
	int BLOCKS = (int) ceil((float)PIXELS / THREADS_PER_BLOCK);

	std::cout << "Blocks: " << BLOCKS << std::endl;
	std::cout << "Threads per block: " << THREADS_PER_BLOCK << std::endl;

	frame.deltax = (frame.winr - frame.winl) / (double) (frame.resx - 1);
	frame.deltay = (frame.wint - frame.winb) / (double) (frame.resy - 1);

	long *out;
	cudaMallocManaged(&out, PIXELS*sizeof(long));

	std::cout << "Starting GPU Compute" << std::endl;

	pixelcalc<<<BLOCKS, THREADS_PER_BLOCK>>>(out, frame);

	cudaDeviceSynchronize();

	std::cout << "Finished GPU Compute" << std::endl;

	for(int x = 0; x < PIXELS; x++)
	{
		out2[x] = out[x];
	}

	cudaFree(out);
}
