#pragma once


class gpu
{
public:
	__host__ static void mandelbrot(long*, long, int, int);
private:
	__host__ __device__ static void pixelcalc(int, int, double, double, double, double, long, long*);
};
