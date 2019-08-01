#include "gpu.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

__host__ __device__
void gpu::pixelcalc(int resx, int resy, double startx, double starty, double deltax, double deltay, long iters, long* out)
{
	int threadnum = blockIdx.x * blockDim.x + threadIdx.x;

	int x = threadnum % resx;
	int y = threadnum / resx;

	if(x > resx || y > resy)
	{
		return;
	}

	double cx = startx + deltax * x;
	double cy = starty + deltay * y;

	double zx = 0.0;
	double zy = 0.0;

	double zx2 = 0.0;
	double zy2 = 0.0;

	double zxt, zyt;

	//out[threadnum] = cx;
	//return;

	for(long i = 0; i < iters; i++)
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

	out[threadnum] = iters;
	return;
}

__host__
void gpu::mandelbrot(long* out2, long iters, int resx, int resy)
{
	long ITERS = iters;

	int RES_X = resx;
	int RES_Y = resy;
	int PIXELS = RES_X * RES_Y;
	int THREADS_PER_BLOCK = 128;
	int BLOCKS = (int) ceil((float)PIXELS / THREADS_PER_BLOCK);

	std::cout << "Blocks: " << BLOCKS << std::endl;
	std::cout << "Threads per block: " << THREADS_PER_BLOCK << std::endl;

	double WIN_L = -2.0;
	double WIN_R = 2.0;
	double WIN_B = -2.0;
	double WIN_T = 2.0;

	double DELTAX = (WIN_R - WIN_L) / (double) (RES_X - 1);
	double DELTAY = (WIN_T - WIN_B) / (double) (RES_Y - 1);

	long *out;
	cudaMallocManaged(&out, PIXELS*sizeof(long));

	std::cout << "Starting GPU Compute" << std::endl;

	pixelcalc<<<BLOCKS, THREADS_PER_BLOCK>>>(RES_X, RES_Y, WIN_L, WIN_B, DELTAX, DELTAY, ITERS, out);

	cudaDeviceSynchronize();

	std::cout << "Finished GPU Compute" << std::endl;

	for(int x = 0; x < PIXELS; x++)
	{
		out2[x] = out[x];
	}

	cudaFree(out);
}
