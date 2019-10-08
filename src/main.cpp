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

//UI
#include <ncurses.h>

//My code
#include "frameinfo.h"
#include "gpu.h"
//#include "fp.h"

#include <unistd.h>
#include "cuda_profiler_api.h"

const int CHARS_PER_PIXEL = 3;

int main(int argc, char **argv)
{
//	fp yee(1, 1);
//	yee.zero();
//	std::cout << yee.toString() << std::endl;

//	return 0;

	if(argc != 2)
	{
		std::cout << "Parameter needed" << std::endl;
		return 1;
	}

	char *arg = argv[1];

	if(strncmp(arg, "i", 1) == 0)
	{
		interactive();
	}
	else if(strncmp(arg, "r", 1) == 0)
	{
		headless();
	}
	else
	{
		std::cout << "i or r" << std::endl;
	}

	return 0;
}

void renderToNC(unsigned long* img, frameinfo frame)
{
	unsigned long val;

	for(int x = 0; x < frame.resx; x++)
	{
		for(int y = 0; y < frame.resy; y++)
		{
			val = img[y * frame.resx + x];

//			attron(COLOR_PAIR(val % 5 + 2));
//			mvprintw(y, x, "%d", val % 10);

			if(val == frame.iters)
			{
				attron(COLOR_PAIR(1));
				mvprintw(y, x, "%d", val % 10);
			}
			else
			{
				attron(COLOR_PAIR(val % 5 + 2));
				mvprintw(y, x, "%d", val % 10);
			}
		}
	}

	attron(COLOR_PAIR(1));
	mvprintw(0, 0, "res=%dx%d", frame.resx, frame.resy);
	mvprintw(1, 0, "center=%fx%f", frame.centerx, frame.centery);
	mvprintw(2, 0, "scale=%e", frame.scale);
	mvprintw(3, 0, "iters=%d", frame.iters);

	refresh();
}

void interactive()
{
	initscr();
	cbreak();
	noecho();

//	WINDOW * win = newwin(LINES, COLS, 0, 0);

	frameinfo frame;
	frame.resx = COLS;
	frame.resy = LINES;
	frame.iters = 1000;

	frame.centerx = 0;
	frame.centery = 0;
	frame.scale = 2;

	unsigned long *img;
	cudaMallocManaged(&img, frame.resx * frame.resy * sizeof(unsigned long));
	gpu::mandelbrot(img, frame);

	start_color();
	init_pair(1, COLOR_WHITE, COLOR_BLACK);
	init_pair(2, COLOR_WHITE, COLOR_RED);
	init_pair(3, COLOR_WHITE, COLOR_YELLOW);
	init_pair(4, COLOR_WHITE, COLOR_GREEN);
	init_pair(5, COLOR_WHITE, COLOR_BLUE);
	init_pair(6, COLOR_WHITE, COLOR_MAGENTA);
	renderToNC(img, frame);

//	mvwprintw(win, 0, 0, "res: %dx%d", frame.resx, frame.resy);
//	wrefresh(win);

	int ch;
	bool running = true;

	while(running)
	{
		ch = getch();
		switch(ch)
		{
			case 'q':
				running = false;
				break;
			case 'a':
				frame.centerx -= frame.scale / 10;
				gpu::mandelbrot(img, frame);
				renderToNC(img, frame);
				break;
			case 'd':
				frame.centerx += frame.scale / 10;
				gpu::mandelbrot(img, frame);
				renderToNC(img, frame);
				break;
			case 'w':
				frame.centery -= frame.scale / 10;
				gpu::mandelbrot(img, frame);
				renderToNC(img, frame);
				break;
			case 's':
				frame.centery += frame.scale / 10;
				gpu::mandelbrot(img, frame);
				renderToNC(img, frame);
				break;
			case 'r':
				frame.scale /= 1.5;
				gpu::mandelbrot(img, frame);
				renderToNC(img, frame);
				break;
			case 'f':
				frame.scale *= 1.5;
				frame.scale = (frame.scale > 2) ? 2 : frame.scale;
				gpu::mandelbrot(img, frame);
				renderToNC(img, frame);
				break;
			case 't':
				frame.iters *= 10;
				gpu::mandelbrot(img, frame);
				renderToNC(img, frame);
				break;
			case 'g':
				frame.iters /= 10;
				gpu::mandelbrot(img, frame);
				renderToNC(img, frame);
				break;
		}
	}

	cudaFree(img);

	endwin();
}

void headless()
{
	frameinfo frame;
	frame.resx = 1920;
	frame.resy = 1080;

	unsigned char *colormap = getColorMap(32);

	unsigned long *out;
	cudaMallocManaged(&out, frame.resx * frame.resy * sizeof(unsigned long));

	const double FPS = 60;
	const double ZOOM_RATE = 0.3333;

	for(int i = 0; i < 1 * FPS; i++)
	{
		frame.centerx = 0;
		frame.centery = -1;

		frame.scale = 2 / pow(10, i * ZOOM_RATE / FPS);
		frame.iters = 100000;

		std::cout << "Rendering Frame #" << i << ", scale=" << frame.scale << std::endl;
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


