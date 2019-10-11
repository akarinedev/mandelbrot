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

//My code
#include "frameinfo.h"

//Libs
#include "json.hpp"

#include "getopt.h"

const int CHARS_PER_PIXEL = 3;

static struct option long_options[] =
{
	{"l", required_argument, NULL, 'l'},
	{"x", required_argument, NULL, 'x'},
	{"y", required_argument, NULL, 'y'},
//	{"framerate", required_argument, NULL, 'r'},
//	{"cx", required_argument, NULL, 'u'},
//	{"cy", required_argument, NULL, 'v'},
//	{"zmstart", optional_argument, NULL, 's'},
//	{"zmend", optional_argument, NULL, 'e'},
//	{"zmrate", optional_argument, NULL, 'z'},
};

int main(int argc, char **argv)
{
	int c;

	int index = 0;

	long length;
	long sx, sy;
	int framerate;
	double cx, cy;
	double zstart, zend, zrate;

	while((c = getopt_long_only(argc, argv, "", long_options, &index)) != -1)
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
		case 'e':
			zend = std::stod(optarg);
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

	for(long i = 0; i < length; i++)
	{
		data[i]["resx"] = sx;
		data[i]["resy"] = sy;
	}

	// write prettified JSON to another file
	//std::ofstream ofstream(file);
	std::cout << std::setw(4) << data << std::endl;

	return 0;
}

