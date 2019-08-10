#include "fp.h"

#include <cstdlib>
#include <stdio.h>
#include <sstream>

fp::fp(int w, int f)
{
	numw = w;
	numf = f;
	numb = w + f;
	data = (uint64_t*) malloc(numb * sizeof(uint64_t));
}

fp::~fp()
{
	free(data);
}

void fp::zero()
{
	for(int i = 0; i < numb; i++)
	{
		data[i] = (uint64_t) 0;
	}
}

std::string fp::toString()
{
	std::stringstream stream;
	char buffer[16];
	for(int i = 0; i < numb; i++)
	{
		sprintf(buffer, "%016lx", data[i]);
		stream << buffer;
		if(i+1 == numw)
		{
			stream << ".";
		}
		else if(i+1 != numb)
		{
			stream << ",";
		}
	}
	std::string out;
	stream >> out;
	return out;
}

void fp::operator=(fp b)
{
	if(numw != b.numw || numf != b.numf)
	{
		throw std::invalid_argument("Mismatch in part numbers");
	}

	for(int i = 0; i < numb; i++)
	{
		data[i] = b.data[i];
	}
}

fp fp::operator+=(fp b)
{
	if(numw != b.numw || numf != b.numf)
	{
		throw std::invalid_argument("Mismatch in part numbers");
	}

	int flag = 0;

	for(int i = numb - 1; i >= 0; i++)
	{
		data[i] += b.data[i] + flag;
		flag = (data[i] < b.data[i]) ? 1 : 0;
	}
}
