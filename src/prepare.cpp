#include "prepare.h"

//Standard Libs
#include <cstdlib>
#include <cmath>

//I/O Streams
#include <iostream>
#include <fstream>
#include <iomanip>

//String stuff
#include <string>
#include <stdio.h>

//Libs
#include "json.hpp"
#include "getopt.h"

#include "gpu.h"

const int CHARS_PER_PIXEL = 3;

static struct option long_options[] =
{
	{"l", optional_argument, NULL, 'l'},
	{"x", optional_argument, NULL, 'x'},
	{"y", optional_argument, NULL, 'y'},
	{"framerate", optional_argument, NULL, 'r'},
	{"cx", optional_argument, NULL, 'u'},
	{"cy", optional_argument, NULL, 'v'},
	{"zstart", optional_argument, NULL, 's'},
	{"zrate", optional_argument, NULL, 'z'}
};

int main(int argc, char **argv)
{
	int c;

	int index = 0;

	long length = 10;
	long sx = 1920, sy = 1080;
	int framerate = 60;
	double cx = 0, cy = -1;
	double zstart = 2, zrate = 0.3333;

	while((c = getopt_long_only(argc, argv, "l:x:y:r:u:v:s:z:", long_options, &index)) != -1)
	{
		switch(c)
		{
		case 'l':
			length = std::stol(optarg);
			break;
		case 'x':
			sx = std::stol(optarg);
			break;
		case 'y':
			sy = std::stol(optarg);
			break;
		case 'r':
			framerate = std::stoi(optarg);
			break;
		case 'u':
			cx = std::stod(optarg);
			break;
		case 'v':
			cy = std::stod(optarg);
			break;
		case 's':
			zstart = std::stod(optarg);
			break;
		case 'z':
			zrate = std::stod(optarg);
			break;
		case ':':
		case '?':
			std::cout << "Error in inputs" << std::endl;
			return 1;
		}
	}

	nlohmann::json data;

	data["length"] = length;
	data["framerate"] = framerate;

	for(long i = 0; i < length; i++)
	{
		data["frames"][i]["resx"] = sx;
		data["frames"][i]["resy"] = sy;
		data["frames"][i]["cx"] = cx;
		data["frames"][i]["cy"] = cy;
		data["frames"][i]["z"] = zstart / pow(10, i*zrate/framerate);
	}

	// write prettified JSON to another file
	//std::ofstream ofstream(file);
	std::cout << std::setw(4) << data << std::endl;

	return 0;
}

