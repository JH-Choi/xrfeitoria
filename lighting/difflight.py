import os
import cv2
import numpy as np
import shutil
import sys
from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from diffusionlight.inpaint import inpaint_chrome_ball
from diffusionlight.ball2envmap import convert_ball2envmap
from diffusionlight.exposure2hdr import convert_ev2hdr
from equirectRotate import EquirectRotate, pointRotate


def rotate_equirectangular_image(image_path: str, c2w: np.ndarray):
    src_image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    h, w, c = src_image.shape

    # calculate relative rotation matrix
    R = c2w[:3, :3]
    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    new_R = np.array([z, -x, -y])
    # rotation_matrix = new_R.T  # same as np.linalg.inv(R)
    rotation_matrix = new_R

    # rotate source image
    equirectRot1 = EquirectRotate(h, w, (0, 0, 0), rotation_matrix=rotation_matrix)
    rotated_image = equirectRot1.rotate(src_image)

    # save .exr image
    output_path = image_path.split('.')[0] + "_rotate.exr"
    cv2.imwrite(output_path, rotated_image)

    return output_path


def get_envmap_from_single_view(image_path, output_dir, c2w=None):
    # step 1: inpaint chrome ball
    # current_folder = os.path.dirname(os.path.abspath(__file__))
    # inpaint_output_dir = os.path.join(current_folder, "diffusionlight/inpaint_output")
    inpaint_output_dir = output_dir
    os.makedirs(inpaint_output_dir, exist_ok=True)
    inpaint_chrome_ball(image_path, inpaint_output_dir)
    # step 2: convert inpainted chrome ball to envmap for each exposure
    square_output_dir = os.path.join(inpaint_output_dir, "square")
    envmap_output_dir = os.path.join(inpaint_output_dir, "envmap")
    convert_ball2envmap(square_output_dir, envmap_output_dir)
    # step 3: convert envmap from each exposure to a single HDR envmap
    hdr_output_dir = os.path.join(inpaint_output_dir, "hdr")
    convert_ev2hdr(envmap_output_dir, hdr_output_dir)
    # step 4: (optional) transform hdr envmap from camera coordinate to world coordinate
    if c2w is not None:
        filename = os.path.basename(image_path).split(".")[0]
        hdr_map_path = os.path.join(hdr_output_dir, filename + ".exr")
        rotated_hdr_map_path = rotate_equirectangular_image(hdr_map_path, c2w)
    # step 5: copy the HDR envmap to the output directory
    os.makedirs(output_dir, exist_ok=True)
    hdr_map_path = shutil.copy(hdr_map_path, output_dir)
    if c2w is not None:
        rotated_hdr_map_path = shutil.copy(rotated_hdr_map_path, output_dir)
    return hdr_output_dir if c2w is None else rotated_hdr_map_path


def get_sunlight_direction(img_path, c2w):
    image = Image.open(img_path).convert('L')
    # image = image.filter(ImageFilter.GaussianBlur(3))
    image = np.array(image)
    max_index = np.unravel_index(np.argmax(image), image.shape)   # Find the index of the maximum intensity value
    y, x = max_index                                              # max_index will contain the (y, x) coordinates of the pixel with the highest intensity
    h, w = image.shape
    theta = (x / w) * 2 * np.pi                                   # convert to spherical coordinates
    phi = (y / h) * np.pi
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    dir_vector = np.array([x, y, z])
    dir_vector = dir_vector / np.linalg.norm(dir_vector)
    dir_vector = c2w[:3, :3] @ dir_vector                         # rotate the direction vector to the world coordinate
    dir_vector = dir_vector / np.linalg.norm(dir_vector)
    dir_vector = -dir_vector
    return dir_vector

# hdr_env_map_path = get_envmap_from_single_view(image_path, output_dir, c2w)
# ev_image_path = os.path.join(output_dir, 'envmap', '{0:05d}_ev-50.png'.format(self.anchor_frame_idx))
# sun_dir = self.get_sunlight_direction(ev_image_path, c2w)
# print('Sunlight direction: ', sun_dir)