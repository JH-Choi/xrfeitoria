import cv2
import numpy as np 
import pdb

def load_rgba(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    pdb.set_trace()
    cv2.imwrite('test.png', image[:, :, 3])
    cv2.imwrite('test2.png', image[:, :, :3])
    # if image.shape[2] == 3:
    #     b, g, r = cv2.split(image)
    #     a = np.ones_like(b) * 255
    #     image = cv2.merge([b, g, r, a])
    # return image


load_rgba('/mnt/hdd/code/human_data_generation/xrfeitoria/output/waypoint_hdr_plane/altitude2.0_offsetX0_offsetY0_scale0.125/img/0000static_camera_0.png')