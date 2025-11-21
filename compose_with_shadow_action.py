import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import OpenEXR
import Imath
import json
import cv2
import torch
import numpy as np
from typing import List
from pathlib import Path
from argparse import ArgumentParser
from video_utils.video_utils import create_video
import pdb
from collections import defaultdict
from torchvision.models.video import (
    MViT_V1_B_Weights,
    MViT_V2_S_Weights,
    R3D_18_Weights,
    S3D_Weights,
    Swin3D_B_Weights,
    Swin3D_T_Weights,
    mvit_v1_b,
    mvit_v2_s,
    r3d_18,
    s3d,
    swin3d_b,
    swin3d_t,
)

model_name_to_model_and_weights = {
    "s3d": (s3d, S3D_Weights.DEFAULT),
    "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT),
    "swin3d_t": (swin3d_t, Swin3D_T_Weights.DEFAULT),
    "swin3d_b": (swin3d_b, Swin3D_B_Weights.DEFAULT),
    "mvit_v1_b": (mvit_v1_b, MViT_V1_B_Weights.DEFAULT),
    "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights.DEFAULT),
}


def crop_and_pad(frame, box, margin_percent):
    """
    Crop box with margin and take square crop from frame.
    """
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1

    # Add margin
    margin_x, margin_y = int(w * margin_percent / 100), int(h * margin_percent / 100)
    x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
    x2, y2 = min(frame.shape[1], x2 + margin_x), min(frame.shape[0], y2 + margin_y)

    # Take square crop from frame
    # size = max(y2 - y1, x2 - x1)
    # center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
    # half_size = size // 2
    # square_crop = frame[
    #     max(0, center_y - half_size) : min(frame.shape[0], center_y + half_size),
    #     max(0, center_x - half_size) : min(frame.shape[1], center_x + half_size),
    # ]
    # return cv2.resize(square_crop, (224, 224), interpolation=cv2.INTER_LINEAR)

    rectangle_crop = frame[y1:y2, x1:x2]
    return cv2.resize(rectangle_crop, (224, 224), interpolation=cv2.INTER_LINEAR)
    # return rectangle_crop


def preprocess_crops_for_video_cls(crops: List[np.ndarray], 
                                    input_size: list = None, 
                                    weights: torch.Tensor = None) -> torch.Tensor:
    # Preprocess a list of crops for video classification.
    # Args:
    #     crops (List[np.ndarray]): List of crops to preprocess. Each crop should have dimensions (H, W, C)
    #     input_size (tuple, optional): The target input size for the model. Defaults to (224, 224).
    # Returns:
    #     torch.Tensor: Preprocessed crops as a tensor with dimensions (1, T, C, H, W).
    if input_size is None:
        input_size = [224, 224]
    from torchvision.transforms import v2

    transform = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(input_size, antialias=True),
            v2.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ]
    )
    processed_crops = [transform(torch.from_numpy(crop).permute(2, 0, 1)) for crop in crops]
    return torch.stack(processed_crops).unsqueeze(0).permute(0, 2, 1, 3, 4)


def create_label_folders(out_path, labels):
    print(f"Creating label folders in {out_path}")
    for label in labels:
        output_folder = out_path / label
        output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


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

    out_action_folder = out_folder / 'composite_w_shadow/actions'
    out_action_folder.mkdir(parents=True, exist_ok=True)
    labels = ["standing","sitting","walking","running","lying"]
    create_label_folders(out_action_folder, labels)
    debug_folder = out_action_folder / 'debug_vis'
    debug_folder.mkdir(parents=True, exist_ok=True)
    debug_vis = True

    ## Video classification parameters
    num_video_sequence_samples = 16
    crop_margin_percentage = 5
    video_cls_overlap_ratio = 0.25 # "overlap ratio between video sequences"
    video_classifier_model = "r3d_18"
    _, weights = model_name_to_model_and_weights[video_classifier_model]
    lbl_dict = {0: "running", 1: "walking", 2: "lying", 3: "sitting", 4: "standing"}

    out_img_folder = out_folder / 'composite_w_shadow/images'

    frame_dict = dict()


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

                if out_idx not in frame_dict:
                    frame_dict[out_idx] = dict()
                    frame_dict[out_idx]['track_ids'] = []
                    frame_dict[out_idx]['bboxs'] = []
                    frame_dict[out_idx]['actions'] = []

                # Composite images
                mask = read_exr(str(mask_path))

                inst_mask = mask * 255
                fg_mask = np.zeros_like(inst_mask)

                lbl_to_tid = dict()
                for tid, lbl in enumerate(label2stencil.keys()):
                    lbl_to_tid[lbl] = tid

                for stencil, lbl in label2stencil.items():
                    non_zero_indices = np.argwhere(inst_mask == int(stencil))
                    if len(non_zero_indices) == 0:
                        continue # this lbel is not present in the mask
                    x_coords, y_coords = non_zero_indices[:, 1], non_zero_indices[:, 0]
                    # Compute min and max values
                    xmin, ymin = np.min(x_coords), np.min(y_coords)
                    xmax, ymax = np.max(x_coords), np.max(y_coords)
 
                    frame_dict[out_idx]['track_ids'].append(lbl_to_tid[stencil])
                    frame_dict[out_idx]['bboxs'].append([xmin, ymin, xmax, ymax])
                    frame_dict[out_idx]['actions'].append(lbl)

    if args.moving_background:
        for cam_idx in range(moving_num_cameras):
            if cam_idx in args.moving_bg_split:
                continue
            start_idx = cam_idx * num_images_per_camera
            for img_idx in range(1, moving_num_images):
                out_idx += 1
                mask_path = mask_folder / sub_folder_name / f'{img_idx:04d}.exr'

                if out_idx not in frame_dict:
                    frame_dict[out_idx] = dict()
                    frame_dict[out_idx]['track_ids'] = []
                    frame_dict[out_idx]['bboxs'] = []
                    frame_dict[out_idx]['actions'] = []

                # Composite images
                mask = read_exr(str(mask_path))

                inst_mask = mask * 255
                fg_mask = np.zeros_like(inst_mask)
                for stencil, lbl in label2stencil.items():
                    non_zero_indices = np.argwhere(inst_mask == int(stencil))
                    if len(non_zero_indices) == 0:
                        continue # this lbel is not present in the mask

                    x_coords, y_coords = non_zero_indices[:, 1], non_zero_indices[:, 0]
                    # Compute min and max values
                    xmin, ymin = np.min(x_coords), np.min(y_coords)
                    xmax, ymax = np.max(x_coords), np.max(y_coords)
 
                    frame_dict[out_idx]['track_ids'].append(lbl_to_tid[stencil])
                    frame_dict[out_idx]['bboxs'].append([xmin, ymin, xmax, ymax])
                    frame_dict[out_idx]['actions'].append(lbl)


    print(" Generate Action Dataset")
    track_history = defaultdict(list)
    # image_files = [file for file in source_folder.iterdir() if file.is_file() and (file.suffix.lower() in ['.jpg', '.png', '.jpeg'])]
    # sorted_image_files = sorted(image_files, key=lambda x: int(x.name[:-4]))
    frame_counter = 0
    tracklet_num = 0 
    for frame_idx in frame_dict:
        frame_path = out_img_folder / f"{frame_idx:05d}.jpg"
        frame = cv2.imread(str(frame_path))

        # frame_counter += 1
        print(f"Processing frame {frame_counter}")

        boxes = frame_dict[frame_idx]['bboxs']
        track_ids = frame_dict[frame_idx]['track_ids']
        actions = frame_dict[frame_idx]['actions']

        crops_to_infer = []
        track_ids_to_infer = []
        actions_to_infer = []

        for box, track_id, action in zip(boxes, track_ids, actions):
            # frame.shape = (2160, 3840, 3) / box.shape = (4,)
            crop = crop_and_pad(frame, box, crop_margin_percentage)
            # crop.shape = (224, 224, 3)
            if debug_vis:
                debug_path = debug_folder / f"{frame_idx:d}_{track_id:d}.jpg"
                cv2.imwrite(str(debug_path), crop)

            track_history[track_id].append(crop)

            if len(track_history[track_id]) > num_video_sequence_samples:
                track_history[track_id].pop(0)

            if len(track_history[track_id]) == num_video_sequence_samples:
                crops = preprocess_crops_for_video_cls(track_history[track_id], weights=weights)
                crops_to_infer.append(crops)
                track_ids_to_infer.append(track_id)
                actions_to_infer.append(action)

        if crops_to_infer and (
            frame_counter % int(num_video_sequence_samples * (1 - video_cls_overlap_ratio)) == 0
        ):
            # crops_to_infer[0].shape = 1,3,16,224,224
            crops_batch = torch.cat(crops_to_infer, dim=0) # crops_batch.shape = 2,3,16,224,224
            print(f"crops_batch shape: {crops_batch.shape}")    
            print(f"track_ids_to_infer: {track_ids_to_infer}")
            print(f"actions_to_infer: {actions_to_infer}")

            for idx in range(len(track_ids_to_infer)):
                tid = track_ids_to_infer[idx]
                gt = actions_to_infer[idx]
                crop = crops_batch[idx]
                # _label_folder = gt.strip().lower()[1:-1]
                _label_folder = lbl_dict[gt]
                if _label_folder not in labels:
                    continue
                _label_path = out_action_folder / _label_folder
                _crop_path = _label_path / f"{tracklet_num:d}_{tid:d}.pt"
                torch.save(crop.clone(), _crop_path)
                tracklet_num += 1

        frame_counter += 1

    import pdb; pdb.set_trace()


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

