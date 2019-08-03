#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <cstdlib>

#include "gpu.h"

int main()
{
	frameinfo frame;

	frame.winl = -2;
	frame.winr = 2;
	frame.winb = -2;
	frame.wint = 2;

	frame.resx = 100;
	frame.resy = 100;

	frame.iters = 1000;

	long *out = (long*) malloc(frame.resx * frame.resy * sizeof(long));

	gpu::mandelbrot(out, frame);

	std::ofstream image;
	image.open("out.pgm");
	image << "P3" << std::endl;
	image << frame.resx << " " << frame.resy << std::endl;
	image << 255 << std::endl;

	std::string colors[12] = {
		"255 0 0", "255 127 0", "255 255 0", "127 255 0",
		"0 255 0", "0 255 127", "0 255 255", "0 127 255",
		"0 0 255", "127 0 255", "255 0 255", "255 0 127"
	};

	long output;

	for(int y = frame.resy - 1; y >= 0; y--)
	{
		for(int x = 0; x < frame.resx; x++)
		{
			output = out[y * frame.resy + x];
			if(output == 0)
			{
				image << "255 255 255\t";
			}
			else if(output == frame.iters)
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
