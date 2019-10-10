#!/bin/sh
ffmpeg -s 1920x1080 -framerate 60 -i imgs/%08d.tga -vcodec libx264 -crf 10 movs/mandelbrot.mp4
