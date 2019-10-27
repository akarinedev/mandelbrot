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

	#ifdef OPT_PTX

	long iters;
	long maxiters = frame.iters;

	asm(
		".reg .u64 i;"
		".reg .f64 cx;"
		".reg .f64 cy;"
		".reg .f64 x;"
		".reg .f64 y;"
//		".reg .f64 xt;"
//		".reg .f64 yt;"
		".reg .f64 x2;"
		".reg .f64 y2;"
		".reg .f64 val;"
		".reg .pred p;"
		"mov.u64 i, 0;"
		"mov.f64 cx, %2;"
		"mov.f64 cy, %3;"
		"mov.f64 x, 0.0;"
		"mov.f64 y, 0.0;"
		"mov.f64 x2, 0.0;"
		"mov.f64 y2, 0.0;"
		"loop:"
			//Values
			//yt = x * y
			"mul.f64 y, x, y;"
			//yt = 2 * yt or yt + yt
			"add.f64 y, y, y;"
			//yt = yt + cy
			"add.f64 y, y, cy;"
			//xt = x^2 - y^2
			"sub.f64 x, x2, y2;"
			//xt = xt + cx
			"add.f64 x, x, cx;"

			//Moves
//			"mov.f64 x, xt;"
//			"mov.f64 y, yt;"

			//Squares
			"mul.f64 x2, x, x;"
			"mul.f64 y2, y, y;"

			//If greater than 4
			"add.f64 val, x2, y2;"
			"setp.ge.f64 p, val, 4.0;"
			"@p bra done;"

			"setp.ge.u64 p, i, %1;"
			"@p bra default;"
			//Increment counter
			"add.u64 i, i, 1;"

			"bra loop;"
		"default:"
		"mov.u64 i, %1;"
		"done:"
		"mov.u64 %0, i;"
		:
		"=l"(iters) :
		"l"(maxiters),
		"d"(cx),
		"d"(cy)
	);

	out[threadnum] = iters;

	#else

	double zx = 0.0;
	double zy = 0.0;

	double zx2 = 0.0;
	double zy2 = 0.0;

	double zxt, zyt;

	#ifdef OPT_SQR

	for(long i = 0; i < frame.iters; i++)
	{
		zy = (zx + zy);
		zy *= zy;
		zy -= zx2;
		zy -= zy2;
		zy += cy;
		zx = zx2 - zy2 + cx;
		zx2 = zx * zx;
		zy2 = zy * zy;

		#ifdef OPT_BULB
		if(i % 20 == 0)
		{
			double p = (zx-.25) * (zx-.25) + zy2;
			double sqrp = sqrt(p);
			if(zx > (sqrp - 2 * p + .25))
			{
				out[threadnum] = frame.iters;
				return;
			}
			if((zx+1)*(zx+1) + zy2 > .0625)
			{
				out[threadnum] = frame.iters;
				return;
			}
		}
		#endif

		if(zx2 + zy2 >= 4)
		{
			out[threadnum] = i;
			return;
		}
	}
	#else
	for(long i = 0; i < frame.iters; i++)
	{
		zxt = zx2 - zy2 + cx;
		zyt = 2 * zx * zy + cy;

		zx = zxt;
		zy = zyt;

		zx2 = zx * zx;
		zy2 = zy * zy;

		#ifdef OPT_BULB
		if(i % 20 == 0)
		{
			double q = (zx - .25) * (zx - .25) + zy2;
			double q2 = q * (q + x - .25);
			if(q2 <= .25 * zy2)
			{
				out[threadnum] = frame.iters;
				return;
			}
		}
		#endif

		if(zx2 + zy2 >= 4)
		{
			out[threadnum] = i;
			return;
		}
	}
	#endif

	out[threadnum] = frame.iters;
	return;

	#endif
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

__host__
unsigned long gpu::itercount(frameinfo frame)
{
	unsigned long *arr1;
	cudaMallocManaged(&arr1, frame.resx * frame.resy * sizeof(unsigned long));
	unsigned long *arr2;
	cudaMallocManaged(&arr2, frame.resx * frame.resy * sizeof(unsigned long));

	unsigned long iters = 1;

	for(unsigned long i = 0; i < frame.resx * frame.resy; i++)
	{
		arr1[i] = 0;
	}

	unsigned long diffs = 0;

	while(true)
	{
		iters *= 2;
		frame.iters = iters;
		gpu::mandelbrot(arr2, frame);

		for(unsigned long i = 0; i < frame.resx * frame.resy; i++)
		{
			if(arr2[i] == iters - 1)
			{
				arr2[i] = 0;
			}
			if(arr1[i] != arr2[i])
			{
				diffs++;
			}
			arr1[i] = arr2[i];
		}

		if(((double) diffs) / frame.resx * frame.resy <= 0.01)
		{
			cudaFree(arr1);
			cudaFree(arr2);
			return iters;
		}
	}
}
