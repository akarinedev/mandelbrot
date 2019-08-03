!/bin/sh
ffmpeg -r 10 -i %08d.png -vcodec libx264 mandelbrot.mp4
