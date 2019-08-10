#pragma once

#include <cstdint>
#include <string>

class fp
{
public:
	int numw;
	int numf;
	int numb;
	uint64_t* data;

	fp(int, int);
	~fp();
	void zero();
	std::string toString();
	void operator=(fp);
	fp operator+=(fp b);
};
