!/bin/sh
ffmpeg -r 30 -i %08d.bmp -vcodec libx264 mandelbrot.mp4
