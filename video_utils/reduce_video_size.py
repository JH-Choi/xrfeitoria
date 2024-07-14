import os
import ffmpeg

# PATH='/mnt/hdd/code/ARLproject/NeRFAugment/ICRA2024'
PATH='/mnt/hdd/code/ARLproject/NeRFAugment'
# input_file = os.path.join(PATH, 'BBox_from_masks.mp4')  # Replace with the path to your input video
# output_file = os.path.join(PATH,'BBox_from_masks_reduced.mp4')  # Replace with the desired output path
input_file = os.path.join(PATH, 'rendered_image.mp4')  # Replace with the path to your input video
output_file = os.path.join(PATH,'rendered_image_reduced.mp4')  # Replace with the desired output path
# input_file = os.path.join(PATH, 'train_15_40_val_50_40_combined.avi')  # Replace with the path to your input video
# output_file = os.path.join(PATH,'train_15_40_val_50_40_combined_reduced.mp4')  # Replace with the desired output path



# Set the video codec and desired output options (e.g., lower bitrate)
ffmpeg.input(input_file).output(output_file, vf='scale=640:480', acodec='aac', vcodec='libx264', crf=28).run()
