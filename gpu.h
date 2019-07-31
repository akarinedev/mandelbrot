#pragma once


class gpu
{
public:
	__host__ static void mandelbrot(long*, long, int, int);
private:
	__global__ static void pixelcalc(int, int, double, double, double, double, long, long*);
};
