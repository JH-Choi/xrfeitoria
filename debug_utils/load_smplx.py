
# https://github.com/vchoutas/smplx/blob/c63c02b478c5c6f696491ed9167e3af6b08d89b1/examples/demo.py
# python load_smplx.py --model-folder /mnt/hdd/data/smplx_data/models_smplx_v1_1/models

import os.path as osp
import argparse

import numpy as np
import torch

# import pyrender
import trimesh
import smplx
import pdb


def main(model_folder, model_type='smplx', ext='npz',
         gender='neutral', plot_joints=False,
         use_face_contour=False):

    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         ext=ext)
    print(model)
    pdb.set_trace()

    betas = torch.randn([1, 10], dtype=torch.float32)
    expression = torch.randn([1, 10], dtype=torch.float32)

    output = model(betas=betas, expression=expression,
                   return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    # pdb.set_trace()
    # vertices *= 0.1
    # tri_mesh = trimesh.Trimesh(vertices, model.faces,
    #                            vertex_colors=vertex_colors)

    # tri_mesh.export('SMPLX_FEMALE.obj')

    # mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    # scene = pyrender.Scene()
    # scene.add(mesh)

    # if plot_joints:
    #     sm = trimesh.creation.uv_sphere(radius=0.005)
    #     sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    #     tfs = np.tile(np.eye(4), (len(joints), 1, 1))
    #     tfs[:, :3, 3] = joints
    #     joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    #     scene.add(joints_pcl)

    # pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    parser.add_argument('--model-folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--model-type', default='smplx', type=str,
                        choices=['smpl', 'smplh', 'smplx'],
                        help='The type of model to load')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='The gender of the model (male, neutral, female)')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--plot-joints', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='The path to the model folder')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')

    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    plot_joints = args.plot_joints
    use_face_contour = args.use_face_contour
    gender = args.gender
    ext = args.ext

    main(model_folder, model_type, ext=ext,
         gender=gender, plot_joints=plot_joints,
         use_face_contour=use_face_contour)