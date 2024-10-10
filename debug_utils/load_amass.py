import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np 
import transformations
import pdb

from xrfeitoria.utils.anim.motion import SMPLMotion, SMPLXMotion

input_amass_smpl_x_path = "/mnt/hdd/data/AMASS/SMPL-X_N/ACCAD/Male2Running_c3d/C3_-_run_stageii.npz"
is_smplx = True
amass_smpl_x_data = np.load(input_amass_smpl_x_path, allow_pickle=True)


if is_smplx:
    src_motion = SMPLXMotion.from_amass_data(amass_smpl_x_data, insert_rest_pose=True)
else:
    src_motion = SMPLMotion.from_amass_data(amass_smpl_x_data, insert_rest_pose=True)
