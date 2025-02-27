import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import OpenEXR
import Imath
import json
import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from video_utils.video_utils import create_video
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
    if args.moving_background:
        # moving_background_folder = out_folder / 'moving_background'
        moving_background_folder = str(out_folder / 'background_moving_camera_{}')

    out_img_folder = out_folder / 'composite_w_shadow/images'
    out_shadow_folder = out_folder / 'composite_w_shadow/shadow'
    out_lbl_folder = out_folder / 'composite_w_shadow/labels'
    out_mask_folder = out_folder / 'composite_w_shadow/masks'
    out_img_folder.mkdir(parents=True, exist_ok=True)
    out_shadow_folder.mkdir(parents=True, exist_ok=True)
    out_mask_folder.mkdir(parents=True, exist_ok=True)
    out_lbl_folder.mkdir(parents=True, exist_ok=True)


    # label to stencil json file
    lbl_to_stencil_file = str(out_folder / 'lbl_stencil.json')
    with open(lbl_to_stencil_file, 'r') as json_file:
        label2stencil = json.load(json_file)

    sub_dirs =  [f for f in foreground_folder.glob('static*') if f.is_dir()]
    sub_dirs = sorted(sub_dirs, key=lambda x: int(str(x).split('/')[-1].split('_')[-1]))
    num_cameras = len(sub_dirs)

    num_images_per_camera = len(list(sub_dirs[0].glob('*.png')))

    if args.moving_background:
        moving_sub_dirs =  [f for f in foreground_folder.glob('moving*') if f.is_dir()]
        moving_num_cameras = len(moving_sub_dirs)
        moving_num_images = len(list(moving_sub_dirs[0].glob('*.png')))

    cam_info = {}
    out_idx = 0 
    start_idx, end_idx = 0, 0   

    if args.static_camera: 
        for cam_idx in range(num_cameras):
            start_idx = cam_idx * num_images_per_camera
            for img_idx in range(1, num_images_per_camera):
                sub_folder_name = sub_dirs[cam_idx].name
                # remove first image since it consists of T-pose humans 
                out_idx = cam_idx * num_images_per_camera + img_idx
                mask_path = mask_folder / sub_folder_name / f'{img_idx:04d}.exr'
                fg_img_path = foreground_folder / sub_folder_name / f'{img_idx:04d}.png'
                bg_img_path = background_folder / f'{cam_idx:05d}.png'
                out_img_path = out_img_folder / f'{out_idx:05d}.jpg'
                out_shadow_path = out_shadow_folder / f'{out_idx:05d}.jpg'
                out_lbl_path = out_lbl_folder / f'{out_idx:05d}.txt'
                out_mask_path = out_mask_folder / f'{out_idx:05d}.jpg'

                # Composite images
                mask = read_exr(str(mask_path))

                inst_mask = mask * 255
                fg_mask = np.zeros_like(inst_mask)
                for stencil, lbl in label2stencil.items():
                    non_zero_indices = np.argwhere(inst_mask == int(stencil))
                    if len(non_zero_indices) == 0:
                        continue # this lbel is not present in the mask
                    fg_mask[non_zero_indices[:, 0], non_zero_indices[:, 1]] = 1
                # cv2.imwrite('fg_mask.png', fg_mask)

                obj_mask = cv2.imread(str(mask_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                obj_mask = obj_mask * 255 > 1

                fg_img = cv2.imread(str(fg_img_path), cv2.IMREAD_UNCHANGED)
                bg_img = cv2.imread(str(bg_img_path), cv2.IMREAD_UNCHANGED)
                # bg_img[obj_mask] = 0

                if args.motion_blur_degree > 0:
                    fg_img = motion_blur(fg_img, degree=args.motion_blur_degree, angle = 45)

                shadow_catcher_alpha = fg_img[:,:,3:] / 255.
                composite_image = bg_img * (1-shadow_catcher_alpha) + fg_img[:,:,0:3] * (shadow_catcher_alpha)
                # composite_image = bg_img + fg_img[:,:,0:3] * (shadow_catcher_alpha)
                cv2.imwrite(str(out_img_path), composite_image)
                cv2.imwrite(str(out_shadow_path), (1 - shadow_catcher_alpha)* 255)

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
                    xmin, ymin = np.min(x_coords), np.min(y_coords)
                    xmax, ymax = np.max(x_coords), np.max(y_coords)
                    debug_img = cv2.rectangle(debug_img,(xmin,ymin),(xmax,ymax),(0,0,255),1) # red    

                    # TODO: comment this line if we want to generate data for action recognition
                    lbl = 0 # person detection 

                    # save bounding box in annotation file, default 0 for pose class, min_x, min_y, width, height
                    # f_annot.write("{} {} {} {} {}\n".format(lbl, xmin, ymin, xmax - xmin, ymax - ymin))

                    # Yolov format
                    WIDTH, HEIGHT = args.resolution
                    x_c = (xmin + ((xmax - xmin) / 2)) / WIDTH
                    y_c = (ymin + ((ymax - ymin) / 2)) / HEIGHT
                    w = (xmax - xmin) / WIDTH
                    h = (ymax - ymin) / HEIGHT
                    f_annot.write("%d %0.6f %0.6f %0.6f %0.6f\n" %(lbl,x_c,y_c,w,h))

                f_annot.close()

                cv2.imwrite(str(out_mask_path), debug_img)
                print(f"Processing camera {cam_idx} / {num_cameras}, image {img_idx} / {num_images_per_camera}, output image {out_idx} / {num_cameras * num_images_per_camera}")

            end_idx = out_idx
            cam_info[cam_idx] = [start_idx, end_idx]

    if args.moving_background:
        for cam_idx in range(moving_num_cameras):
            if cam_idx in args.moving_bg_split:
                continue
            start_idx = cam_idx * num_images_per_camera
            for img_idx in range(1, moving_num_images):
                out_idx += 1
                sub_folder_name = 'moving_camera_{}'.format(cam_idx)
                mask_path = mask_folder / sub_folder_name / f'{img_idx:04d}.exr'
                fg_img_path = foreground_folder / sub_folder_name / f'{img_idx:04d}.png'
                bg_img_path = os.path.join(moving_background_folder.format(cam_idx), f'{img_idx:05d}.png')
                out_img_path = out_img_folder / f'{out_idx:05d}.jpg'
                out_shadow_path = out_shadow_folder / f'{out_idx:05d}.jpg'
                out_lbl_path = out_lbl_folder / f'{out_idx:05d}.txt'
                out_mask_path = out_mask_folder / f'{out_idx:05d}.jpg'

                # Composite images
                mask = read_exr(str(mask_path))

                inst_mask = mask * 255
                fg_mask = np.zeros_like(inst_mask)
                for stencil, lbl in label2stencil.items():
                    non_zero_indices = np.argwhere(inst_mask == int(stencil))
                    if len(non_zero_indices) == 0:
                        continue # this lbel is not present in the mask
                    fg_mask[non_zero_indices[:, 0], non_zero_indices[:, 1]] = 1
                # cv2.imwrite('fg_mask.png', fg_mask)

                obj_mask = cv2.imread(str(mask_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                obj_mask = obj_mask * 255 > 1

                fg_img = cv2.imread(str(fg_img_path), cv2.IMREAD_UNCHANGED)
                bg_img = cv2.imread(str(bg_img_path), cv2.IMREAD_UNCHANGED)
                # bg_img[obj_mask] = 0

                if args.motion_blur_degree > 0:
                    fg_img = motion_blur(fg_img, degree=args.motion_blur_degree, angle = 45)

                shadow_catcher_alpha = fg_img[:,:,3:] / 255.
                composite_image = bg_img * (1-shadow_catcher_alpha) + fg_img[:,:,0:3] * (shadow_catcher_alpha)
                # composite_image = bg_img + fg_img[:,:,0:3] * (shadow_catcher_alpha)
                cv2.imwrite(str(out_img_path), composite_image)
                cv2.imwrite(str(out_shadow_path), (1 - shadow_catcher_alpha)* 255)

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
                    xmin, ymin = np.min(x_coords), np.min(y_coords)
                    xmax, ymax = np.max(x_coords), np.max(y_coords)
                    debug_img = cv2.rectangle(debug_img,(xmin,ymin),(xmax,ymax),(0,0,255),1) # red    

                    # TODO: comment this line if we want to generate data for action recognition
                    lbl = 0 # person detection 

                    # save bounding box in annotation file, default 0 for pose class, min_x, min_y, width, height
                    # f_annot.write("{} {} {} {} {}\n".format(lbl, xmin, ymin, xmax - xmin, ymax - ymin))

                    # Yolov format
                    WIDTH, HEIGHT = args.resolution
                    x_c = (xmin + ((xmax - xmin) / 2)) / WIDTH
                    y_c = (ymin + ((ymax - ymin) / 2)) / HEIGHT
                    w = (xmax - xmin) / WIDTH
                    h = (ymax - ymin) / HEIGHT
                    f_annot.write("%d %0.6f %0.6f %0.6f %0.6f\n" %(lbl,x_c,y_c,w,h))

                f_annot.close()

                cv2.imwrite(str(out_mask_path), debug_img)
                print(f"Processing camera {cam_idx} / {moving_num_cameras}, image {img_idx} / {moving_num_images}, output image {out_idx} / {moving_num_cameras * moving_num_images + num_cameras * num_images_per_camera}")

            end_idx = out_idx
            cam_info[cam_idx] = [start_idx, end_idx]

    with open(str(out_folder / 'composite_w_shadow/cam_info.json'), 'w') as json_file:
        json.dump(cam_info, json_file)

    create_video(str(out_img_folder), str(out_folder / 'composite_image_w_shadow.mp4'))
    create_video(str(out_mask_folder), str(out_folder / 'composite_mask_w_shadow.mp4'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default=None, help='output path')
    parser.add_argument('--motion_blur_degree', type=int, default=4, help='output path')
    parser.add_argument('--resolution', type=int, nargs='+', default=[662, 363], help='resolution')
    parser.add_argument('--static_camera', default=False, action='store_true', help='Moving camera Trajectory')
    parser.add_argument('--moving_background', default=False, action='store_true', help='Moving camera Trajectory')
    parser.add_argument('--moving_bg_split', type=int, nargs='+', default=[], help='skip moving background split')
    args = parser.parse_args()

    main(args)

