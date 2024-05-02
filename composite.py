import OpenEXR
import Imath
import cv2
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
bg_img_path = '/mnt/hdd/code/gaussian_splatting/gaussian_splatting_large/output/Okutama_Noon_AdaptMesh/render_w_pose/ours_30000/00003.png'

mask = read_exr(mask_path)

# fg_img = cv2.imread(fg_img_path, cv2.IMREAD_UNCHANGED)
bg_img = cv2.imread(bg_img_path, cv2.IMREAD_UNCHANGED)
# bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
fg_img = cv2.imread(fg_img_path, cv2.IMREAD_UNCHANGED)


def composite_images(mask, foreground, background):
    mask = mask > 0 
    foreground_mask = foreground[:,:,3]
    # cv2.imwrite('foreground_mask.png', foreground_mask)
    foreground = foreground[:,:,0:3]
    
    # Multiply foreground with normalized mask
    masked_foreground = foreground * mask
    # cv2.imwrite('masked_foreground.png', masked_foreground)
    
    # Invert the mask
    inverted_mask = 1 - mask
    
    # Multiply background with inverted mask
    masked_background = background * inverted_mask
    
    # Add masked foreground and masked background
    composite_image = masked_foreground + masked_background
    
    return composite_image

tot_img = composite_images(mask, fg_img, bg_img)
cv2.imwrite('composite.png', tot_img)