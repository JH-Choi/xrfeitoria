import os
import glob
import json
import argparse
from PIL import Image
from tqdm import tqdm
import pdb

parser = argparse.ArgumentParser(description='sample colmap images')
parser.add_argument('--root_path', type=str, default=None, help='input path')
parser.add_argument('--sub_folders', type=str, nargs='+', default=[], help='data split')
parser.add_argument('--out_path', type=str, default=None, help='output path')
args = parser.parse_args()

ROOT_PATH='/mnt/hdd/code/human_data_generation/xrfeitoria/output'
# REAL_FOLDERS = [] 

############
OUT_PATH='/mnt/hdd/code/human_data_generation/xrfeitoria/output/S2_Noon_v0'
# REAL_FOLDERS=['/mnt/hdd/data/Okutama_Action/yolov8_Detection/1.2.2/'] 
REAL_FOLDERS=[] 
SUB_FOLDERS=[ 
    'S2_Drone1_Noon_1_2_2/auto_Drone1_Noon_1_2_2_alti0',
    'S2_Drone2_Noon_2_2_2/auto_Drone2_Noon_2_2_2_alti0',
    'S2_Drone1_Noon_1_2_4/auto_Drone1_Noon_1_2_4_alti0',
    'S2_Drone2_Noon_2_2_4/auto_Drone2_Noon_2_2_4_alti0',
    'S2_Drone1_Noon_1_2_9/auto_Drone1_Noon_1_2_9_alti0',
    'S2_Drone2_Noon_2_2_9/auto_Drone2_Noon_2_2_9_alti0',
]
############

# OUT_PATH='/mnt/hdd/code/human_data_generation/xrfeitoria/output/S3_Morning_v0'
# # REAL_FOLDERS=['/mnt/hdd/data/Okutama_Action/yolov8_Detection/2.2.4/'] 
# REAL_FOLDERS=[] 
# SUB_FOLDERS=[ 
#     'S3_Drone1_Morning_1_1_1/auto_Drone1_Morning_1_1_1_alti0',
#     'S3_Drone1_Morning_1_1_4/auto_Drone1_Morning_1_1_4_alti0',
#     'S3_Drone1_Morning_1_1_7/auto_Drone1_Morning_1_1_7_alti0',
#     'S3_Drone2_Morning_2_1_1/auto_Drone2_Morning_2_1_1_alti0',
#     'S3_Drone2_Morning_2_1_10/auto_Drone2_Morning_2_1_10_alti0',
# ]
#############



RAW_IMG_TXT = os.path.join(OUT_PATH, 'image_mapping.txt')
CONFIG_INFO = os.path.join(OUT_PATH, 'config_info.json') 

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUT_PATH, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUT_PATH, 'labels'), exist_ok=True)    

f_img_txt = open(RAW_IMG_TXT, 'w')

data_config = dict()

data_idx = 1 
for sub_folder in SUB_FOLDERS:
    REAL_PATH=os.path.join(ROOT_PATH, sub_folder, 'composite_w_shadow')
    real_paths = sorted(glob.glob(REAL_PATH + '/images/*.jpg'))
    real_lbl =  sorted(glob.glob(REAL_PATH + '/labels/*.txt'))
    assert len(real_paths) == len(real_lbl)

    num_images = len(real_paths)
    data_config['[SYN]:' + sub_folder] = num_images
    for i in tqdm(range(num_images)):
        real_img_path = real_paths[i]
        real_lbl_path = real_lbl[i]
        os.system(f'cp -r {real_img_path}  {OUT_PATH}/images/{data_idx:05d}.png')
        os.system(f'cp -r {real_lbl_path}  {OUT_PATH}/labels/{data_idx:05d}.txt')

        f_img_txt.write(f'{data_idx:05d} {real_img_path}\n')
        data_idx += 1

if len(REAL_FOLDERS) > 0:
    for real_folder in REAL_FOLDERS:
        real_paths = sorted(glob.glob(real_folder + '/images/*.jpg'))
        real_lbl =  sorted(glob.glob(real_folder + '/labels/*.txt'))
        assert len(real_paths) == len(real_lbl)

        num_images = len(real_paths)
        data_config['[REAL]:' + real_folder.split('/')[-2]] = num_images
        for i in tqdm(range(num_images)):
            real_img_path = real_paths[i]
            real_lbl_path = real_lbl[i]
            os.system(f'cp -r {real_img_path}  {OUT_PATH}/images/{data_idx:05d}.jpg')
            os.system(f'cp -r {real_lbl_path}  {OUT_PATH}/labels/{data_idx:05d}.txt')

            f_img_txt.write(f'{data_idx:05d} {real_img_path}\n')
            data_idx += 1

json.dump(data_config, open(CONFIG_INFO, 'w'), indent=4) 
