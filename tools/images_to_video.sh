ffmpeg -framerate 15 -pattern_type glob -i '/home/mat/thesis/data/waymo/train/pc_demo_images/*.png' -c:v libx264 -pix_fmt yuv420p '/home/mat/thesis/data/waymo/train/camera.mp4'
