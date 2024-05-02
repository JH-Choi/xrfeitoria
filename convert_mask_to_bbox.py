import os
import cv2
import OpenEXR
import Imath
import numpy as np
import pdb

def read_exr(input_exr):
    # Open the OpenEXR file
    exr_file = OpenEXR.InputFile(input_exr)

    # Get the image header
    header = exr_file.header()

    # Get the image size
    dw = header['displayWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Read the RGB channels
    channels = ['R', 'G', 'B']
    channel_arrays = {}
    for channel in channels:
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        channel_arrays[channel] = np.frombuffer(exr_file.channel(channel, pixel_type), dtype=np.float32)
        channel_arrays[channel] = np.reshape(channel_arrays[channel], (height, width))

    # Combine channels to get the RGB image
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)
    for i, channel in enumerate(channels):
        rgb_image[:, :, i] = channel_arrays[channel]
    return rgb_image


mask_path = './output/blender/alti5_radi0.5_cam6/mask/static_camera_0/0001.exr'
fg_img_path = './output/blender/alti5_radi0.5_cam6/img/static_camera_0/0001.png'
fg_img = cv2.imread(fg_img_path)

mask = read_exr(mask_path) * 255
stencil_list = [50, 100, 150, 200 ,250]

for label in stencil_list:
    non_zero_indices = np.argwhere(mask == label)
    # Extract x and y coordinates
    x_coords, y_coords = non_zero_indices[:, 1], non_zero_indices[:, 0]

    # Compute min and max values
    min_x, min_y = np.min(x_coords), np.min(y_coords)
    max_x, max_y = np.max(x_coords), np.max(y_coords)
    print(f"Min: ({min_x}, {min_y}), Max: ({max_x}, {max_y})")
    fg_img = cv2.rectangle(fg_img,(min_x,min_y),(max_x,max_y),(0,0,255),5) # red    


    # # save bounding box in annotation file, default 0 for pose class, min_x, min_y, width, height
    # f_annot.write("{} {} {} {} {}\n".format(label, min_x, min_y, max_x - min_x, max_y - min_y))

# write out image to bbox_screenshots
cv2.imwrite('tmp.png',fg_img)
# cv2.imwrite(os.path.join(dst_dir,save_name),raw_img)
# f_annot.close()
# pdb.set_trace()