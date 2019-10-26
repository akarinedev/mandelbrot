#include "render.h"

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
//#include "fp.h"

#include <unistd.h>
#include "cuda_profiler_api.h"

#include "json.hpp"

const int CHARS_PER_PIXEL = 3;

int main(int argc, char **argv)
{
	unsigned char *colormap = getColorMap(32);


	nlohmann::json data;
	std::cin >> data;

	long length = data["length"];

	long resx = data["frames"][0]["resx"];
	long resy = data["frames"][0]["resy"];

	unsigned long *out;
	cudaMallocManaged(&out, resx * resy * sizeof(unsigned long));

	for(int i = 0; i < length; i++)
	{
		auto df = data["frames"][i];

		frameinfo frame;
		frame.resx = df["resx"];
		frame.resy = df["resy"];
		frame.centerx = df["cx"];
		frame.centery = df["cy"];
		frame.scale = df["z"];
		frame.iters = 1000;

		std::cout << "Rendering Frame #" << i << " / " << length << ", scale=" << frame.scale << std::endl;
		gpu::mandelbrot(out, frame);

		char name[100];
		snprintf(name, 100, "imgs/%08d.tga", i);

		std::cout << "Saving Image: " << name << std::endl;
		saveImage(name, out, frame, colormap, 32);
	}

	cudaFree(out);

	free(colormap);

	return 0;
}

void saveImage(char *name, unsigned long *in, frameinfo frame, unsigned char *colormap, int numcolors)
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
				color = (output - 1) % numcolors;

				image[coord + 0] = colormap[CHARS_PER_PIXEL * color + 0];
				image[coord + 1] = colormap[CHARS_PER_PIXEL * color + 1];
				image[coord + 2] = colormap[CHARS_PER_PIXEL * color + 2];

				//image[x][y] = png::rgb_pixel(colormap[color][0], colormap[color][1], colormap[color][2]);
			}
		}
	}

	//stbi_write_png(name, resx, resy, CHARS_PER_PIXEL, image, resx * sizeof(unsigned char) * CHARS_PER_PIXEL);
	stbi_write_tga(name, resx, resy, CHARS_PER_PIXEL, image);

	free(image);
}

unsigned char *getColorMap(int numcolors)
{
	unsigned char *colormap = (unsigned char*) malloc(numcolors * CHARS_PER_PIXEL * sizeof(unsigned char));

	double deltaangle = (float) 360 / numcolors;
	for(int i = 0; i < numcolors; i++)
	{
		double h = deltaangle * i / 60;
		double x = (1 - std::abs(std::fmod(h, 2) - 1)) * 255;

		if(h <= 1){
			colormap[CHARS_PER_PIXEL * i + 0] = 255; colormap[CHARS_PER_PIXEL * i + 1] = x; colormap[CHARS_PER_PIXEL * i + 2] = 0;}
		else if(h <= 2){
			colormap[CHARS_PER_PIXEL * i + 0] = x; colormap[CHARS_PER_PIXEL * i + 1] = 255; colormap[CHARS_PER_PIXEL * i + 2] = 0;}
		else if(h <= 3){
			colormap[CHARS_PER_PIXEL * i + 0] = 0; colormap[CHARS_PER_PIXEL * i + 1] = 255; colormap[CHARS_PER_PIXEL * i + 2] = x;}
		else if(h <= 4){
			colormap[CHARS_PER_PIXEL * i + 0] = 0; colormap[CHARS_PER_PIXEL * i + 1] = x; colormap[CHARS_PER_PIXEL * i + 2] = 255;}
		else if(h <= 5){
			colormap[CHARS_PER_PIXEL * i + 0] = x; colormap[CHARS_PER_PIXEL * i + 1] = 0; colormap[CHARS_PER_PIXEL * i + 2] = 255;}
		else if(h <= 6){
			colormap[CHARS_PER_PIXEL * i + 0] = 255; colormap[CHARS_PER_PIXEL * i + 1] = 0; colormap[CHARS_PER_PIXEL * i + 2] = x;}
	}

	return colormap;
}


