ffmpeg -framerate 15 -pattern_type glob -i 'val_drawn/*.png' \
  -c:v libx264 -pix_fmt yuv420p out.mp4
