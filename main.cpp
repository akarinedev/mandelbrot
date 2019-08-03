#include <iostream>
#include <cmath>
#include <cstdlib>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

	int colormap[12][3] = {
		{255, 0, 0}, {255, 127, 0}, {255, 255, 0}, {127, 255, 0},
		{0, 255, 0}, {0, 255, 127}, {0, 255, 255}, {0, 127, 255},
		{0, 0, 255}, {127, 0, 255}, {255, 0, 255}, {255, 0, 127}
	};

	//unsigned char *image = stbi_load("mandelbrot.png", frame.resx, frame.resy, 3, 0);

	unsigned char *image = (unsigned char*) malloc(frame.resx * frame.resy * sizeof(unsigned char) * 3);

	long x, y, output, coord;
	int color;

	for(x = 0; x < frame.resx; x++)
	{
		for(y = 0; y < frame.resy; y++)
		{
			coord = y * frame.resx * 3 + x * 3;

			output = out[y * frame.resy + x];
			if(output == 0) //Exited after 0 iterations (outside of 2 circle)
			{
				image[coord + 0] = 255;
				image[coord + 1] = 255;
				image[coord + 2] = 255;
				//image[x][y] = png::rgb_pixel(255, 255, 255);
			}
			else if(output == frame.iters)
			{
				image[coord + 0] = 0;
				image[coord + 1] = 0;
				image[coord + 2] = 0;

				//image[x][y] = png::rgb_pixel(0, 0, 0);
			}
			else
			{
				color = (output - 1) % 12;

				image[coord + 0] = colormap[color][0];
				image[coord + 1] = colormap[color][1];
				image[coord + 2] = colormap[color][2];

				//image[x][y] = png::rgb_pixel(colormap[color][0], colormap[color][1], colormap[color][2]);
			}
		}
	}

	stbi_write_png("mandelbrot.png", frame.resx, frame.resy, 3, image, frame.resx * 3 * sizeof(unsigned char));

	free(image);
	free(out);

	return 0;
}
