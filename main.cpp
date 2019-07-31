#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <cstdlib>

#include "gpu.h"

int main()
{
	int RES_X = 100;
	int RES_Y = 100;
	int ITERS = 1000;

	long *out = (long*) malloc(RES_X * RES_Y * sizeof(long));

	gpu::mandelbrot(out, ITERS, RES_X, RES_Y);

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
	free(out);

	return 0;
}
