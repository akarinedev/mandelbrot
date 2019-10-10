#include "interactive.h"

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

const int CHARS_PER_PIXEL = 3;

int main(int argc, char **argv)
{
	interactive();
}

void renderToNC(unsigned long* img, frameinfo frame)
{
	unsigned long val;

	for(int x = 0; x < frame.resx; x++)
	{
		for(int y = 0; y < frame.resy; y++)
		{
			val = img[y * frame.resx + x];

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
