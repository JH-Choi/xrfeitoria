import OpenEXR
import Imath
import json
import cv2
import argparse
import numpy as np
from pathlib import Path
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


# Background image folder
bg_img_folder = Path('/mnt/hdd/code/gaussian_splatting/Octree-GS/outputs/Okutama_Action/Yonghan_data/okutama_n50_Noon/OkutamaMesh_Noon_baseline_baselayer12_wMask/2024-05-03_14:18:30/render_w_pose/ours_40000')

# Foreground image and mask folder
mask_folder = Path('./output/blender_waypoint/MySequence/mask')
fg_img_folder = Path('./output/blender_waypoint/MySequence/img')

# Output folder
out_img_folder = Path('./output/blender_waypoint/MySequence/composite/images')
out_lbl_folder = Path('./output/blender_waypoint/MySequence/composite/labels')
out_mask_folder = Path('./output/blender_waypoint/MySequence/composite/masks')
out_img_folder.mkdir(parents=True, exist_ok=True)
out_mask_folder.mkdir(parents=True, exist_ok=True)
out_lbl_folder.mkdir(parents=True, exist_ok=True)

# label to stencil json file
lbl_to_stencil_file = "./output/blender_waypoint/MySequence/lbl_stencil.json"   

with open(lbl_to_stencil_file, 'r') as json_file:
    label2stencil = json.load(json_file)

num_cameras = 22

sub_dirs =  [f for f in fg_img_folder.glob('*') if f.is_dir()]
assert len(sub_dirs) == num_cameras

num_images_per_camera = len(list(sub_dirs[0].glob('*.png')))


for cam_idx in range(num_cameras):
    for img_idx in range(1, num_images_per_camera):
        # remove first image since it consists of T-pose humans 
        out_idx = cam_idx * num_images_per_camera + img_idx
        mask_path = mask_folder / f'static_camera_{cam_idx}' / f'{img_idx:04d}.exr'
        fg_img_path = fg_img_folder / f'static_camera_{cam_idx}' / f'{img_idx:04d}.png'
        bg_img_path = bg_img_folder / f'{cam_idx:05d}.png'
        out_img_path = out_img_folder / f'{out_idx:05d}.jpg'
        out_lbl_path = out_lbl_folder / f'{out_idx:05d}.txt'
        out_mask_path = out_mask_folder / f'{out_idx:05d}.jpg'

        # Composite images
        mask = read_exr(str(mask_path))
        fg_img = cv2.imread(str(fg_img_path), cv2.IMREAD_UNCHANGED)
        bg_img = cv2.imread(str(bg_img_path), cv2.IMREAD_UNCHANGED)
        tot_img = composite_images(mask, fg_img, bg_img)
        cv2.imwrite(str(out_img_path), tot_img)

        # Generate Bounding Box Labels
        debug_img = cv2.imread(str(out_img_path), cv2.IMREAD_UNCHANGED)

        f_annot = open(out_lbl_path, 'w')
        mask = mask * 255
        for stencil, lbl in label2stencil.items():
            non_zero_indices = np.argwhere(mask == int(stencil))
            if len(non_zero_indices) == 0:
                continue # this lbel is not present in the mask
            x_coords, y_coords = non_zero_indices[:, 1], non_zero_indices[:, 0]

            # Compute min and max values
            min_x, min_y = np.min(x_coords), np.min(y_coords)
            max_x, max_y = np.max(x_coords), np.max(y_coords)
            # tot_img = cv2.rectangle(tot_img,(min_x,min_y),(max_x,max_y),(0,0,255),5) # red    
            debug_img = cv2.rectangle(debug_img,(min_x,min_y),(max_x,max_y),(0,0,255),5) # red    

            # save bounding box in annotation file, default 0 for pose class, min_x, min_y, width, height
            f_annot.write("{} {} {} {} {}\n".format(lbl, min_x, min_y, max_x - min_x, max_y - min_y))
        f_annot.close()

        cv2.imwrite(str(out_mask_path), debug_img)
        print(f"Processing camera {cam_idx} / {num_cameras}, image {img_idx} / {num_images_per_camera}, output image {out_idx} / {num_cameras * num_images_per_camera}")