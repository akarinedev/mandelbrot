#include "main.h"

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
#include "getopt.h"
#include "cuda_profiler_api.h"

const int CHARS_PER_PIXEL = 3;

static struct option long_options[] =
{
	{"x", required_argument, NULL, 'x'},
	{"y", required_argument, NULL, 'y'},
	{"framerate", required_argument, NULL, 'r'},
	{"cx", required_argument, NULL, 'u'},
	{"cy", required_argument, NULL, 'v'},
	{"zmstart", optional_argument, NULL, 's'},
	{"zmend", optional_argument, NULL, 'e'},
	{"zmrate", optional_argument, NULL, 'z'}
};

int main(int argc, char **argv)
{
	int c;

	int index = 0;

	long sx, sy;
	int framerate;
	double cx, cy;
	double zstart, zend, zrate;

	while((c = getopt_long_only(argc, argv, "", long_options, &index)) != -1)
	{
		switch(c)
		{
		case 'x':
			sx = std::stol(optarg);
			break;
		case 'y':
			sy = std::stol(optarg);
			break;
		case 'r':
			framerate = std::stoi(optarg);
		case 'u':
			cx = std::stod(optarg);
		case 'v':
			cy = std::stod(optarg);
		case 's':
			zstart = std::stod(optarg);
		case 'e':
			zend = std::stod(optarg);
		case 'z':
			zrate = std::stod(optarg);

		}
	}

	std::cout << "X: " << sx << std::endl;

	return 0;
}

void headless()
{
	frameinfo frame;
	frame.resx = 1920;
	frame.resy = 1080;

	unsigned char *colormap = getColorMap(32);

	unsigned long *out;
	cudaMallocManaged(&out, frame.resx * frame.resy * sizeof(unsigned long));

	const double VID_LENGTH = 1; //40
	const double FPS = 2; //30
	const double ZOOM_PER_SECOND = 0.25; //.333
	const double ZOOM_PER_FRAME = ZOOM_PER_SECOND / FPS;

	for(int i = 0; i < VID_LENGTH * FPS; i++)
	{
		frame.centerx = 0;
		frame.centery = -1;

		frame.scale = 2 / pow(10, i * ZOOM_PER_FRAME);
		frame.iters = 100000;

		std::cout << "Rendering Frame #" << i << " / " << VID_LENGTH * FPS << ", scale=" << frame.scale << std::endl;
		gpu::mandelbrot(out, frame);

		char name[] = "100";
		snprintf(name, 100, "imgs/%08d.tga", i);

		std::cout << "Saving Image: " << name << std::endl;
		saveImage(name, out, frame, colormap, 32);
	}

	cudaFree(out);

	free(colormap);
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


