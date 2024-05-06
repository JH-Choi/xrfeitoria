import OpenEXR
import Imath
import cv2
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


mask_folder = Path('./output/blender_waypoint/MySequence/mask')
fg_img_folder = Path('./output/blender_waypoint/MySequence/img')
bg_img_folder = Path('/mnt/hdd/code/gaussian_splatting/Octree-GS/outputs/Okutama_Action/Yonghan_data/okutama_n50_Noon/OkutamaMesh_Noon_baseline_baselayer12_wMask/2024-05-03_14:18:30/render_w_pose/ours_40000')
out_folder = Path('./output/blender_waypoint/MySequence/composite')
out_folder.mkdir(parents=True, exist_ok=True)
num_cameras = 12

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
        out_img_path = out_folder / f'{out_idx:05d}.png'

        mask = read_exr(str(mask_path))
        fg_img = cv2.imread(str(fg_img_path), cv2.IMREAD_UNCHANGED)
        bg_img = cv2.imread(str(bg_img_path), cv2.IMREAD_UNCHANGED)

        tot_img = composite_images(mask, fg_img, bg_img)
        cv2.imwrite(str(out_img_path), tot_img)
        print(f'Saved {out_img_path}')
        pdb.set_trace()

pdb.set_trace()

# mask = read_exr(mask_path)
# # fg_img = cv2.imread(fg_img_path, cv2.IMREAD_UNCHANGED)
# bg_img = cv2.imread(bg_img_path, cv2.IMREAD_UNCHANGED)
# # bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
# fg_img = cv2.imread(fg_img_path, cv2.IMREAD_UNCHANGED)

# tot_img = composite_images(mask, fg_img, bg_img)
# # cv2.imwrite('composite.png', tot_img)