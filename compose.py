import os
import OpenEXR
import Imath
import json
import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
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


 
def motion_blur(image, degree=12, angle=45):
    """
    degree : intensity of blur
    angle : direction of blur. 
        angle = 0 is +u, angle = 90 is -v
    """
    image = np.array(image)
    angle -= 135  # because np.diag create a 135 degree rotated kernel
 
    # a matrix of motion blur kernels at any angle is generated. 
    # The larger the degree, the higher the level of blur.
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
 
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred



def main(args):
    out_folder = Path(args.output_path)
    background_folder = out_folder / 'background'
    mask_folder = out_folder / 'mask'
    foreground_folder = out_folder / 'img'

    out_img_folder = out_folder / 'composite/images'
    out_lbl_folder = out_folder / 'composite/labels'
    out_mask_folder = out_folder / 'composite/masks'
    out_img_folder.mkdir(parents=True, exist_ok=True)
    out_mask_folder.mkdir(parents=True, exist_ok=True)
    out_lbl_folder.mkdir(parents=True, exist_ok=True)


    # label to stencil json file
    lbl_to_stencil_file = str(out_folder / 'lbl_stencil.json')
    with open(lbl_to_stencil_file, 'r') as json_file:
        label2stencil = json.load(json_file)

    sub_dirs =  [f for f in foreground_folder.glob('*') if f.is_dir()]
    num_cameras = len(sub_dirs)

    num_images_per_camera = len(list(sub_dirs[0].glob('*.png')))

    for cam_idx in range(num_cameras):
        for img_idx in range(1, num_images_per_camera):
            # remove first image since it consists of T-pose humans 
            out_idx = cam_idx * num_images_per_camera + img_idx
            mask_path = mask_folder / f'static_camera_{cam_idx}' / f'{img_idx:04d}.exr'
            fg_img_path = foreground_folder / f'static_camera_{cam_idx}' / f'{img_idx:04d}.png'
            bg_img_path = background_folder / f'{cam_idx:05d}.png'
            out_img_path = out_img_folder / f'{out_idx:05d}.jpg'
            out_lbl_path = out_lbl_folder / f'{out_idx:05d}.txt'
            out_mask_path = out_mask_folder / f'{out_idx:05d}.jpg'

            # Composite images
            mask = read_exr(str(mask_path))
            fg_img = cv2.imread(str(fg_img_path), cv2.IMREAD_UNCHANGED)
            bg_img = cv2.imread(str(bg_img_path), cv2.IMREAD_UNCHANGED)

            fg_img = motion_blur(fg_img, degree=args.motion_blur_degree, angle = 45)

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

                # TODO: comment this line if we want to generate data for action recognition
                lbl = 0 # person detection 

                # save bounding box in annotation file, default 0 for pose class, min_x, min_y, width, height
                f_annot.write("{} {} {} {} {}\n".format(lbl, min_x, min_y, max_x - min_x, max_y - min_y))
            f_annot.close()

            cv2.imwrite(str(out_mask_path), debug_img)
            print(f"Processing camera {cam_idx} / {num_cameras}, image {img_idx} / {num_images_per_camera}, output image {out_idx} / {num_cameras * num_images_per_camera}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default=None, help='output path')
    parser.add_argument('--motion_blur_degree', type=int, default=4, help='output path')
    args = parser.parse_args()

    main(args)
