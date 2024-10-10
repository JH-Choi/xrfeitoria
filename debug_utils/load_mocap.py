# Code from Ml-hugs
# https://github.com/apple/ml-neuman/issues/66

import os
import torch
import numpy as np 
from smpl import SMPL
import transformations
import pdb


# def get_manual_alignment(opt):
def get_manual_alignment(scene_name):
    # if os.path.basename(opt.scene_dir) == 'bike' and opt.motion_name == 'jumpandroll':
    #     manual_trans = np.array([0.08, 0.12, 0.4])
    #     manual_rot = np.array([95.8, 10.4, 1.8]) / 180 * np.pi
    #     manual_scale = 0.14
    # else:
    if True:
        manual_trans = np.array([0, 0, 0])
        manual_rot = np.array([0, 0, 0]) / 180 * np.pi
        manual_scale = 1
    return manual_trans, manual_rot, manual_scale

def to_homogeneous(pts):
    if isinstance(pts, torch.Tensor):
        return torch.cat([pts, torch.ones_like(pts[..., 0:1])], axis=-1)
    elif isinstance(pts, np.ndarray):
        return np.concatenate([pts, np.ones_like(pts[..., 0:1])], axis=-1)


# motion_path = "/mnt/hdd/data/AMASS/SMPL-X_N/ACCAD/Male2Running_c3d/C3_-_run_stageii.npz"
motion_path = "/mnt/hdd/data/AMASS/SMPL-H_G/SFU/0012/0012_JumpAndRoll001_poses.npz"
motions = np.load(motion_path, allow_pickle=True)

for key in motions.keys():
    print(key)
    print(motions[key].shape)

poses = motions['poses']
poses = poses[:, :72]
poses[:, 66:] = 0
# trans = motions['trans'][start_idx:end_idx:skip]
trans = motions['trans']
# beta = scene.smpls[0]['betas']

n_frames = trans.shape[0]
beta = motions['betas'][:10]

body_model = SMPL(
    '/mnt/hdd/code/gaussian_splatting/ml-hugs/data/smpl/',
    gender='neutral',
    device=torch.device('cpu')
)


# read manual alignment
manual_trans, manual_rot, manual_scale = get_manual_alignment(None)
M_R = transformations.euler_matrix(*manual_rot)
M_S = np.eye(4)
M_S[:3, :3] *= manual_scale
M_T = transformations.translation_matrix(manual_trans)
T_mocap2scene = M_T[None] @ M_S[None] @ M_R[None]


# 大 pose
da_smpl = np.zeros_like(np.zeros((1, 72)))
da_smpl = da_smpl.reshape(-1, 3)
da_smpl[1] = np.array([0, 0, 1.0])
da_smpl[2] = np.array([0, 0, -1.0])
da_smpl = da_smpl.reshape(1, -1)


raw_verts = []
Ts = []
for i, p in enumerate(poses):
    import pdb; pdb.set_trace()
    # transformations from T-pose to mocap pose(random scale)
    _, T_t2mocap = body_model.verts_transformations(
        return_tensor=False,
        poses=p[None],
        betas=beta[None],
        transl=trans[i][None]
    )
    # transform mocap data to scene space
    T_t2scene = T_mocap2scene @ T_t2mocap
    # T-pose to Da-pose
    _, T_t2da = body_model.verts_transformations(
        return_tensor=False,
        poses=da_smpl,
        betas=beta[None]
    )
    # Da-pose to scene space
    T_da2scene = T_t2scene @ np.linalg.inv(T_t2da)
    # Da-pose verts
    temp_static_verts, _ = body_model(
        return_tensor=False,
        return_joints=True,
        poses=da_smpl,
        betas=beta[None]
    )
    # verts in scene
    verts = np.einsum('BNi, Bi->BN', T_da2scene, to_homogeneous(temp_static_verts))[:, :3].astype(np.float32)
    raw_verts.append(verts)
    Ts.append(T_da2scene)

pdb.set_trace()

# SMPLX => trans, root_orient, poses 


#### ML-hugs
# global_orient, body_pose, betas, transl, smpl_scale