import os
import mediapy as media
import glob
import cv2
from contextlib import ExitStack

# if output_format == "video":
#                     if writer is None:
#                         render_width = int(render_image.shape[1])
#                         render_height = int(render_image.shape[0])
#                         writer = stack.enter_context(
#                             media.VideoWriter(
#                                 path=output_filename,
#                                 shape=(render_height, render_width),
#                                 fps=fps,
#                             )
#                         )
#                     writer.add_image(render_image)


# data_dir = './output/blender_waypoint/altitude1.0/composite/images'
data_dir = './output/blender_waypoint/altitude1.0/composite/masks'
render_width = 1280  
render_height = 720
seconds = 20
output_filename = 'altitude1.0_v2_mask.mp4'


image_files = sorted(glob.glob(os.path.join(data_dir, '*.jpg')), key=lambda x: int(os.path.basename(x).split('.')[0]))
# image_files = sorted(glob.glob(os.path.join(data_dir, '*.png')), key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
# image_files = image_files[:200]
image_files = image_files[-200:]
length = len(image_files)
fps = length / seconds
print('fps', fps)

with ExitStack() as stack:
    writer = None
    for image_file in image_files:
        if writer is None:
            writer = stack.enter_context(media.VideoWriter(
                path=output_filename,
                shape=(render_height, render_width),
                fps=fps,
            ))
        print(image_file)
        image = cv2.imread(image_file)
        render_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        writer.add_image(render_image)
