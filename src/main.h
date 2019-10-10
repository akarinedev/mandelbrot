#pragma once

#include "frameinfo.h"

int main(int, char**);

void headless();

unsigned char *getColorMap(int);
void saveImage(char*, unsigned long*, frameinfo, unsigned char*, int);


