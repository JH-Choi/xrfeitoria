import os
import re
import math
import json
import glob
import random
import numpy as np 
import xrfeitoria as xf
from argparse import ArgumentParser
from pathlib import Path
from xrfeitoria.data_structure.models import RenderPass
from xrfeitoria.data_structure.models import SequenceTransformKey as SeqTransKey
from xrfeitoria.rpc import remote_blender
from xrfeitoria.utils.anim import load_amass_motion

from scipy.spatial.transform import Rotation as R
from xrfeitoria.utils.colmap_utils import read_model, convert_to_blender_coord, qvec2rotmat, rotmat2qvec
from xrfeitoria.utils.camera_utils import quaternion_slerp, focal2fov, get_rotation_matrix
from xrfeitoria.utils.motion_utils import generate_plane, read_motion, apply_scale
from video_utils.traj import generate_interpolated_path

import pdb

##### Okutama Action
def extract_sort_key(filename):
    """Extracts numerical components from a filename and returns them as a tuple of integers."""
    parts = re.findall(r'\d+', filename) 
    return tuple(map(int, parts))  # Convert to tuple for proper sorting

def generate_random_points(x_min, x_max, y_min, y_max, z, N):
    points = [(random.uniform(x_min, x_max), random.uniform(y_min, y_max), z) for _ in range(N)]
    return points

def random_partition(N, parts=5):
    # Generate 4 random breakpoints in range [1, N-1]
    breakpoints = sorted(random.sample(range(1, N), parts - 1))
    # Compute the differences between consecutive breakpoints
    numbers = [breakpoints[0]] + [breakpoints[i] - breakpoints[i-1] for i in range(1, parts-1)] + [N - breakpoints[-1]]
    return numbers

def main(args):
    os.makedirs(args.output_path, exist_ok=True)
   
    # randomly sample origins
    print('area xcoord , ', args.area_xcoord)
    print('area ycoord , ', args.area_ycoord)
    x_min, x_max = args.area_xcoord
    y_min, y_max = args.area_ycoord
    origin_list = generate_random_points(x_min, x_max, y_min, y_max, args.zcoord, \
                                            args.num_actors)

    # Human file 
    actor_template_path = Path(args.actor_template_path)
    # actor_template_list = [d for d in actor_template_path.iterdir() if d.is_dir()]
    human_wear_black_pools = [6001,6043,6045,6083,6085,6120,6145,6152,6216,6232,6237,6174,6200] * 2
    actor_template_list = [actor_template_path / f"{number:07d}"  for number in human_wear_black_pools]
    actor_list = random.sample(actor_template_list, args.num_actors)
    human_list = [str(fbx / 'SMPL-XL-baked.fbx') for fbx in actor_list]

    # Action labels
    lbl_dict = {"running": 0, "walking": 1, "lying":2,  "sitting": 3, "standing": 4}
    stencil_list = [int(stencil) for stencil in np.linspace(10, 255, args.num_actors)]
    assert len(human_list) == len(stencil_list)

    lying_action_file=os.path.join(args.motion_path, 'ACCAD/Male1General_c3d/General_A9_-__Lie_Down_stageii.npz')
    standing_action_file=os.path.join(args.motion_path, 'ACCAD/Male2General_c3d/A1-_Stand_stageii.npz')
    sitting_action_file=os.path.join(args.motion_path, 'BMLrub/rub001/0009_sitting1_stageii.npz')
    walking_motion_files = glob.glob(os.path.join(args.motion_path, 'ACCAD', '**Running_c3d', '*.npz'))
    running_motion_files = glob.glob(os.path.join(args.motion_path, 'ACCAD', '**Walking_c3d', '*.npz'))
    walking_motion_files = [f for f in walking_motion_files if 'neutral_stage' not in f]
    running_motion_files = [f for f in running_motion_files if 'neutral_stage' not in f]

    if args.random_partition:
        label_nums = random_partition(args.num_actors, parts=len(lbl_dict.keys()))
    else:
        label_nums = [int(args.num_actors * ratio) for ratio in args.actor_parition_ratio]

    motion_list, lbl_list = [], []
    min_frame_num = np.inf
    for lbl_idx, sub_label_num in enumerate(label_nums):
        if lbl_idx == 0:  # running
            motion_name = "running"
            sub_motion_files = np.random.choice(running_motion_files, sub_label_num)
            print('running motion files ', sub_motion_files)
        elif lbl_idx == 1: # walking
            motion_name = "walking"
            sub_motion_files = np.random.choice(walking_motion_files, sub_label_num)
            print('walking motion files ', sub_motion_files)
        elif lbl_idx == 2: # lying
            motion_name = "lying"
            sub_motion_files = [lying_action_file for _ in range(sub_label_num)]
        elif lbl_idx == 3: # sitting
            motion_name = "sitting"
            sub_motion_files = [sitting_action_file for _ in range(sub_label_num)]
        elif lbl_idx == 4:# standing
            motion_name = "standing"
            sub_motion_files = [standing_action_file for _ in range(sub_label_num)]
        
        for motion_file in sub_motion_files:
            lbl_list.append(lbl_dict[motion_name])
            if lbl_idx == 3:
                anim_motion = read_motion(motion_file, fps=args.fps, start_frame=78, end_frame=135)
            elif lbl_idx == 0 or lbl_idx == 1:
                anim_motion = read_motion(motion_file, fps=args.fps)
                if anim_motion[1] < min_frame_num :
                    min_frame_num = anim_motion[1]
            else:
                anim_motion = read_motion(motion_file, fps=args.fps)
            motion_list.append(anim_motion)

    assert len(motion_list) == len(origin_list) == len(human_list)

    xf_runner = xf.init_blender(exec_path=args.engine_exec_path, 
                                background=args.background, 
                                new_process=True)

    # Load Plane for shadow catcher
    if args.use_plane:
        plane_size = 20
        xf_runner.Shape.spawn_plane(name='plane', location=(0,0, args.zcoord))
        generate_plane('plane', size=plane_size)

    # Load Background Current mesh 
    if args.background_mesh_file is not None:
        xf_runner.utils.import_file(file_path=args.background_mesh_file)
        print('Load background mesh')

    # Load Colmap data
    root_path = Path(args.colmap_path)
    cameras, images, points3D = read_model(str(root_path), ext='.bin')

    # select only split for visualiation
    if args.split is not None:
        new_images = {}
        for key, val in images.items():
            if args.split not in val.name:
                continue
            else:
                new_images[key] = val
        images = new_images

    colmap_data = {}
    colmap_data['cameras'] = cameras
    colmap_data['images'] = images

    intrinsic_param = np.array([camera.params for camera in colmap_data['cameras'].values()])
    image_quaternion = np.stack([img.qvec for img in colmap_data['images'].values()])
    image_translation = np.stack([img.tvec for img in colmap_data['images'].values()])
    image_names = np.stack([img.name for img in colmap_data['images'].values()])
    # sort_image_id = np.argsort(image_names)

    sorted_indices_and_names = sorted(enumerate(image_names), key=lambda x: extract_sort_key(x[1]))
    sort_image_id = [idx for idx, _ in sorted_indices_and_names]
    # sorted_filenames = [name for _, name in sorted_indices_and_names]

    if args.moving_camera:
        camtoworlds_all = []
        transform_keys = []
        tot_quats, tot_trans = [], []
        tot_Location_moving, tot_Rotation_moving = [], []
        for idx in range(len(sort_image_id)):
            curr_idx = sort_image_id[idx]
            tvec_cur = image_translation[curr_idx]
            qvec_cur = image_quaternion[curr_idx]
            R_ = np.transpose(qvec2rotmat(qvec_cur))
            T_ = np.array(tvec_cur)
            Rt = np.zeros((4, 4))
            Rt[:3, :3] = R_.transpose()
            Rt[:3, 3] = T_
            Rt[3, 3] = 1.0
            C2W = np.linalg.inv(Rt)
            camtoworlds_all.append(C2W[:3, :])

        camtoworlds_all = generate_interpolated_path(np.array(camtoworlds_all), args.interpolate_rate)  # [N, 3, 4]
        camtoworlds_all = np.concatenate([camtoworlds_all, np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0)],axis=1)  # [N, 4, 4]
        for C2W in camtoworlds_all:    
            Rt = np.linalg.inv(C2W)
            qvec = rotmat2qvec(Rt[:3, :3])
            tvec = Rt[:3, 3]
            tot_quats.append(qvec)
            tot_trans.append(tvec)

        for idx, (qvec_w2c, tvec_w2c) in enumerate(zip(tot_quats, tot_trans)):
            tvec, qvec = convert_to_blender_coord(tvec_w2c, qvec_w2c)
            location = (tvec[0], tvec[1], tvec[2] + args.altitude)
            # qvec in the order [w,x,y,z]
            qvec = np.array([qvec[1], qvec[2], qvec[3], qvec[0]])
            Rot = R.from_quat(qvec)
            rotation = Rot.as_euler('xyz', degrees=True) 
            rotation = tuple(r for r in rotation) # (156.53, 57.99, 62.94)

            tot_Location_moving.append(location)
            tot_Rotation_moving.append(rotation)

            # Add a transform key to the moving camera
            transform_keys.append(
                SeqTransKey(
                    frame=idx,
                    location=location,
                    rotation=rotation,
                    interpolation='AUTO',
                )
            )  

    # if args.moving_camera:
    #     max_step = 5
    #     diff_image_translation = image_translation[1:] - image_translation[:-1]
    #     length_diff = np.linalg.norm(diff_image_translation, axis=1)
    #     min_length = np.min(length_diff)
    #     step_list = length_diff / min_length
    #     step_list = np.round(step_list, 0).astype(int)
    #     step_list = np.minimum(step_list, max_step)

    #     transform_keys = []
    #     tot_quats, tot_trans = [], []
    #     tot_Location_moving, tot_Rotation_moving = [], []
    #     for idx in range(sort_image_id.shape[0] - 1):
    #         ts = np.linspace(0, 1, step_list[idx])
    #         # ts = np.linspace(0, 1, 10)

    #         curr_idx = sort_image_id[idx]
    #         next_idx = sort_image_id[idx + 1]

    #         tvec_cur = image_translation[curr_idx]
    #         qvec_cur = image_quaternion[curr_idx]

    #         tvec_next = image_translation[next_idx]
    #         qvec_next = image_quaternion[next_idx]

    #         quats = [quaternion_slerp(qvec_cur, qvec_next, t) for t in ts]
    #         trans =  [(1 - t) * tvec_cur + t * tvec_next for t in ts]

    #         tot_quats.extend(quats)
    #         tot_trans.extend(trans)

    #     for idx, (qvec_w2c, tvec_w2c) in enumerate(zip(tot_quats, tot_trans)):
    #         tvec, qvec = convert_to_blender_coord(tvec_w2c, qvec_w2c)
    #         location = (tvec[0], tvec[1], tvec[2] + args.altitude)
    #         # qvec in the order [w,x,y,z]
    #         qvec = np.array([qvec[1], qvec[2], qvec[3], qvec[0]])
    #         Rot = R.from_quat(qvec)
    #         rotation = Rot.as_euler('xyz', degrees=True) 
    #         rotation = tuple(r for r in rotation) # (156.53, 57.99, 62.94)

    #         tot_Location_moving.append(location)
    #         tot_Rotation_moving.append(rotation)

    #         # Add a transform key to the moving camera
    #         transform_keys.append(
    #             SeqTransKey(
    #                 frame=idx,
    #                 location=location,
    #                 rotation=rotation,
    #                 interpolation='AUTO',
    #             )
    #         )  


 
    # save trajectory for rendering Gaussian Splatting
    tot_Rotation, tot_Location = [], []

    with xf_runner.Sequence.new(seq_name=args.sequence_name, seq_length=min_frame_num, replace=True) as seq:
        # Load Actors
        actor_list = []
        for i, (motion_data, actor_path, stentcil_val) in enumerate(zip(motion_list, human_list, stencil_list)):   
            actor_list.append(xf_runner.Actor.import_from_file(file_path=actor_path, stencil_value=stentcil_val))
            apply_scale(actor_list[-1].name, scale_factor=args.actor_scale_factor)  # SMPL-XL model is imported with scale, we need to apply scale to it
            actor_list[-1].location = origin_list[i]
            actor_list[-1].rotation = [0,0,0]
            xf_runner.utils.apply_motion_data_to_actor(motion_data=motion_data[0], actor_name=actor_list[-1].name)

        if args.moving_camera:
            # Add a moving camera rotating around the actors
            moving_camera = seq.spawn_camera_with_keys(
                camera_name=f'moving_camera',
                transform_keys=transform_keys,
                fov=args.fov,
            )

        # Load Cameras 
        for idx, i_id in enumerate(sort_image_id):
            if idx % args.sampling_rate != 0:
                continue
            tvec_w2c = image_translation[i_id]
            qvec_w2c = image_quaternion[i_id]
            tvec, qvec = convert_to_blender_coord(tvec_w2c, qvec_w2c)

            # location = (tvec[0], tvec[1], tvec[2])
            location = (tvec[0], tvec[1], tvec[2] + args.altitude)
            # qvec in the order [w,x,y,z]
            qvec = np.array([qvec[1], qvec[2], qvec[3], qvec[0]])
            Rot = R.from_quat(qvec)
            rotation = Rot.as_euler('xyz', degrees=True)
            rotation = tuple(r for r in rotation)

            tot_Location.append(location)
            tot_Rotation.append(rotation)  

            # Add a static camera
            static_camera = seq.spawn_camera(
                camera_name=f'static_camera_{idx}',
                location=location,
                rotation=rotation,
                fov=args.fov,
            )

            # use the `camera` in level to render
            seq.use_camera(camera=static_camera)

        # Load HDR map
        if args.hdri_path:
            print('Set HDR map {}'.format(args.hdri_path))
            xf_runner.utils.set_hdr_map(hdr_map_path=args.hdri_path)

        # Add a render job to renderer
        # In render job, you can specify the output path, resolution, render passes, etc.
        # The output path is the path to save the rendered data.
        # The resolution is the resolution of the rendered image.
        # The render passes define what kind of data you want to render, such as img, depth, normal, etc.
        # and what kind of format you want to save, such as png, exr, etc.
        seq.add_to_renderer(
            output_path=args.output_path,
            resolution=tuple(args.resolution),
            render_passes=[
                RenderPass('img', 'png'),
                RenderPass('depth', 'exr'),  
                RenderPass('mask', 'exr')
            ], 
            render_samples=args.render_samples,  # default value 128 / fast 32
            transparent_background=True, 
        )
        # RenderPass('img', 'png'),

        # export verts of meshes in this sequence and its level
        # xf_runner.utils.export_vertices(export_path=output_path / seq_2_name / 'vertices')

    xf_runner.render()
    pdb.set_trace()


    # Save the camera trajectory to a json file 
    R_BlenderView_to_OpenCVView = np.diag([1,-1,-1])
    fl_x=args.focal_x
    res_dict = {'R': [], 'T': [], 'fl_x': fl_x}
    for rot_, loc_ in zip(tot_Rotation, tot_Location):
        R_BlenderView = R.from_euler('xyz', rot_, degrees=True).as_matrix()
        T_BlenderView = np.array(loc_)
        R_OpenCV = R_BlenderView_to_OpenCVView @ np.transpose(R_BlenderView)
        T_OpenCV = -1.0 * R_OpenCV @ T_BlenderView
        R_OpenCV = np.transpose(R_OpenCV)
        res_dict['R'].append([[element for element in row] for row in R_OpenCV])
        res_dict['T'].append([row for row in T_OpenCV])

    os.makedirs(os.path.join(args.output_path, args.sequence_name), exist_ok=True)
    cam_file_path = os.path.join(args.output_path, args.sequence_name, 'camera_trajectory.json')
    with open(cam_file_path, "w") as outfile:
        json.dump(res_dict, outfile, indent=True)
    print(f'Results saved to "{cam_file_path}". # of frames: {len(tot_Rotation)}')

    if args.moving_camera:
        res_dict_moving = {'R': [], 'T': [], 'fl_x': fl_x}
        for rot_, loc_ in zip(tot_Rotation_moving, tot_Location_moving):
            R_BlenderView = R.from_euler('xyz', rot_, degrees=True).as_matrix()
            T_BlenderView = np.array(loc_)
            R_OpenCV = R_BlenderView_to_OpenCVView @ np.transpose(R_BlenderView)
            T_OpenCV = -1.0 * R_OpenCV @ T_BlenderView
            R_OpenCV = np.transpose(R_OpenCV)
            res_dict_moving['R'].append([[element for element in row] for row in R_OpenCV])
            res_dict_moving['T'].append([row for row in T_OpenCV])

        os.makedirs(os.path.join(args.output_path, args.sequence_name), exist_ok=True)
        cam_file_path = os.path.join(args.output_path, args.sequence_name, 'moving_camera_trajectory.json')
        with open(cam_file_path, "w") as outfile:
            json.dump(res_dict_moving, outfile, indent=True)
        print(f'Results saved to "{cam_file_path}". # of frames: {len(tot_Rotation_moving)}')

    # Save sten: label
    lbl_stencil_dict ={}
    for lbl, sten in zip(lbl_list, stencil_list):
        lbl_stencil_dict[sten] = lbl
    with open(os.path.join(args.output_path, args.sequence_name, 'lbl_stencil.json'), 'w') as f:
        json.dump(lbl_stencil_dict, f, indent=True)
    f.close()

    # Render
    # xf_runner.render()

    # save all configs to a json file
    with open(os.path.join(args.output_path, args.sequence_name, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=True)
    f.close()
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--engine_exec_path', type=str, 
                        default='/mnt/hdd/code/blender/blender-3.6.9-linux-x64/blender', 
                        help='Path to the blender executable')
    parser.add_argument('--background', default=False, action='store_true', help='Run blender in background')
    parser.add_argument('--background_mesh_file', type=str, default=None, help='Background mesh file')
    parser.add_argument('--colmap_path', type=str, default=None, help='data root folder')
    parser.add_argument('--split', type=str, default=None, help='data split')
    parser.add_argument('--hdri_path', type=str, default=None, help='hdri path')
    parser.add_argument('--output_path', type=str, default=None, help='output path')
    parser.add_argument('--num_actors', type=int, default=12, help='number of actors')
    parser.add_argument('--area_xcoord', type=float, nargs='+', default=[-3, 1], help='resolution')
    parser.add_argument('--area_ycoord', type=float, nargs='+', default=[-1, 4], help='resolution')
    parser.add_argument('--zcoord', type=int, default=-3, help='z coordinate')
    parser.add_argument('--motion_path', type=str, default=None, help='Path to AMASS dataset')
    parser.add_argument('--actor_template_path', type=str, default=None, help='actor template path')
    parser.add_argument('--actor_scale_factor', type=float, default=0.1, help='actor scale factor')
    parser.add_argument('--random_partition', default=False, action='store_true', help='randomly distribute actors')
    parser.add_argument('--actor_parition_ratio', type=float, nargs='+', default=[0.4, 0.4, 0.05, 0.05, 0.1], help='actor partition ratio, [running, walking, lying, sitting, standing]')
    # parser.add_argument('--interpolate_steps', type=int, default=4, help='interpolation steps')
    parser.add_argument('--sequence_name', type=str, default='test', help='sequence name')
    parser.add_argument('--sampling_rate', type=int, default=10, help='sampling camera poses')
    parser.add_argument('--use_plane', default=False, action='store_true', help='Use plane')
    parser.add_argument('--fov', type=int, default=90, help='Field of view')
    parser.add_argument('--focal_x', type=float, default=763.5, help='Focal length')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--resolution', type=int, nargs='+', default=[662, 363], help='resolution')
    parser.add_argument('--render_samples', type=int, default=128, help='fast version:32 / high resolution: 128')
    parser.add_argument('--altitude', type=float, default=0, help='altitude change')
    parser.add_argument('--moving_camera', default=False, action='store_true', help='Moving camera Trajectory')
    parser.add_argument('--interpolate_rate', type=int, default=5, help='interpolation rate')
    args = parser.parse_args()

    main(args)


