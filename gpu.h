#pragma once


namespace gpu
{
	__host__ void mandelbrot(long*, long, int, int);
	__global__ void pixelcalc(int, int, double, double, double, double, long, long*);
};
