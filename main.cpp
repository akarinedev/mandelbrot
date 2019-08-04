//Standard Libs
#include <cstdlib>
#include <cmath>
#include <iostream>

//String stuff
#include <string>
#include <stdio.h>

//Image writing
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//My code
#include "frameinfo.h"
#include "gpu.h"

const int colormap[12][3] = {
	{255, 0, 0}, {255, 127, 0}, {255, 255, 0}, {127, 255, 0},
	{0, 255, 0}, {0, 255, 127}, {0, 255, 255}, {0, 127, 255},
	{0, 0, 255}, {127, 0, 255}, {255, 0, 255}, {255, 0, 127}
};

const int CHARS_PER_PIXEL = 3;

void renderImage(char *name, unsigned long *in, frameinfo frame)
{
	int resx = frame.resx;
	int resy = frame.resy;
	unsigned long iters = frame.iters;

	unsigned char *image = (unsigned char*) malloc(resx * resy * sizeof(unsigned char) * CHARS_PER_PIXEL);

	int x, y;
	unsigned long output;
	long coord;
	int color;

	for(x = 0; x < resx; x++)
	{
		for(y = 0; y < resy; y++)
		{
			coord = y * resx + x;
			output = in[y * resx + x];
			coord *= CHARS_PER_PIXEL;

			if(output == 0) //Exited after 0 iterations (outside of 2 circle)
			{
				image[coord + 0] = 255;
				image[coord + 1] = 255;
				image[coord + 2] = 255;
				//image[x][y] = png::rgb_pixel(255, 255, 255);
			}
			else if(output == iters)
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

	//stbi_write_png(name, resx, resy, CHARS_PER_PIXEL, image, resx * sizeof(unsigned char) * CHARS_PER_PIXEL);
	stbi_write_bmp(name, resx, resy, CHARS_PER_PIXEL, image);

	free(image);
}

int main()
{
	frameinfo frame;
	frame.resx = 1920;
	frame.resy = 1080;

	unsigned long *out;
	cudaMallocManaged(&out, frame.resx * frame.resy * sizeof(unsigned long));

	for(int i = 0; i < 2500; i++)
	{
		frame.centerx = 0;
		frame.centery = -1;

		frame.scale = 2 / pow(10, i/(float)100);
		frame.iters = 1000;

		std::cout << "Starting GPU Compute" << std::endl;
		gpu::mandelbrot(out, frame);
		std::cout << "Finished GPU Compute" << std::endl;

		char name[] = "100";
		snprintf(name, 100, "render/%08d.bmp", i);

		std::cout << "Saving Image: " << name << std::endl;
		renderImage(name, out, frame);
	}

	cudaFree(out);

	return 0;
}
