#pragma once

#include <ncurses.h>

#include "frameinfo.h"

int main(int, char**);
void renderToNC(WINDOW*, unsigned long*, frameinfo);
void interactive();

void headless();

unsigned char *getColorMap(int);
void saveImage(char*, unsigned long*, frameinfo, unsigned char*, int);


