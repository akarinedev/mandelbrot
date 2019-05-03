#include <iostream>
#include <cmath>
#include <fstream>
#include <string>

__global__
void mandelbrot(int resx, int resy, double startx, double starty, double deltax, double deltay, long iters, long* out)
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

int main()
{
	long ITERS = 1000;

	int RES_X = 10000;
	int RES_Y = 10000;
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

	mandelbrot<<<BLOCKS, THREADS_PER_BLOCK>>>(RES_X, RES_Y, WIN_L, WIN_B, DELTAX, DELTAY, ITERS, out);

	cudaDeviceSynchronize();

	std::cout << "Finished GPU Compute" << std::endl;

	std::ofstream image;
	image.open("out.pgm");
	image << "P3" << std::endl;
	image << RES_X << " " << RES_Y << std::endl;
	image << 255 << std::endl;

	std::string colors[12] = {
		"255 0 0", "255 127 0", "255 255 0", "127 255 0",
		"0 255 0", "0 255 127", "0 255 255", "0 127 255",
		"0 0 255", "127 0 255", "255 0 255", "255 0 127"
	};

	long output;

	for(int y = RES_Y - 1; y >= 0; y--)
	{
		for(int x = 0; x < RES_X; x++)
		{
			output = out[y * RES_Y + x];
			if(output == 0)
			{
				image << "255 255 255\t";
			}
			else if(output == ITERS)
			{
				image << "0 0 0\t";
			}
			else
			{
				image << colors[(output - 1) % 12] << "\t";
				
			}
		}

		image << std::endl;
	}

	image.close();
	cudaFree(out);

	return 0;
}
