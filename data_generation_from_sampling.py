import os
import math
import json
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
from xrfeitoria.utils.colmap_utils import read_model, convert_to_blender_coord, qvec2rotmat
from xrfeitoria.utils.camera_utils import quaternion_slerp, focal2fov, get_rotation_matrix

import bpy
import json
import pdb


@remote_blender()
def apply_scale(actor_name: str, scale_factor: float):

    obj = bpy.data.objects.get(actor_name)
    # bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    # bpy.data.objects[actor_name].select_set(True)
    bpy.ops.object.transform_apply(scale=True)
    obj.scale.x *= scale_factor
    obj.scale.y *= scale_factor
    obj.scale.z *= scale_factor


@remote_blender()
def generate_plane(actor_name: str, size):
    plane = bpy.data.objects.get(actor_name)
    plane.select_set(True)
    plane.scale = (size, size, 1)
    plane.is_shadow_catcher = True

def read_motion(motion_path, fps=None, start_frame=None, end_frame=None):
    # fps:  convert the motion from 120fps (amass) to 30fps
    #  cut the motion to 10 frames, for demonstration purpose
    motion = load_amass_motion(motion_path)  # modify this to motion file in absolute path
    if fps is not None:
        motion.convert_fps(fps)  # convert the motion from 120fps (amass) to 30fps
    motion.cut_motion(start_frame=start_frame, end_frame=end_frame)
    motion_data = motion.get_motion_data()
    n_frames = motion.n_frames
    return motion_data, n_frames


def main(args):
    render_engine = 'blender'
    offsetX = 5  # Offset for controlling waypoints, camera orbit, and actor origins
    offsetY = 3  # Offset for controlling waypoints, camera orbit, and actor origins

    xf_runner = xf.init_blender(exec_path=args.engine_exec_path, 
                                background=args.background, 
                                new_process=True)

    # Load Actors
    if args.actor_template_path is not None:
        actor_template_path = Path(args.actor_template_path)
        actor_template_list = [d for d in actor_template_path.iterdir() if d.is_dir()]
        actor_list = random.sample(actor_template_list, args.num_actors)
        human_list = [str(fbx / 'SMPL-XL-baked.fbx') for fbx in actor_list]

    # Load Plane for shadow catcher
    if args.use_plane:
        size = 30
        xf_runner.Shape.spawn_plane(name='plane', location=(0,0,-1.5))
        generate_plane('plane', size=size)

    # Load Background Current mesh 
    if args.background_mesh_file is not None:
        xf_runner.utils.import_file(file_path=args.background_mesh_file)
        print('Load background mesh')

    # Load the motion data from the JSON file
    if args.actor_info_file is not None:
        with open(args.actor_info_file, 'r') as json_file:
            motion_dict = json.load(json_file)

        stencil_list = [int(stencil) for stencil in np.linspace(10, 255, args.num_actors)]
        assert len(human_list) == len(stencil_list)

        # Load Actor motion data
        motion_list, origin_list, rot_list, lbl_list = [], [], [], []
        lbl_dict = {"running": 0, "walking": 1, "lying":2,  "sitting": 3, "standing": 4}
        min_frame_num = np.inf
        for motion_name, motion_info in motion_dict.items():
            print("Motion:", motion_name)
            for motion_path in motion_info["motion_paths"]:
                lbl_list.append(lbl_dict[motion_name])

                if motion_name == 'sitting':
                    # motion_list.append(read_motion(motion_path, fps=fps, start_frame=310, end_frame=540)) # when fps is None
                    anim_motion = read_motion(motion_path, fps=args.fps, start_frame=78, end_frame=135)
                    motion_list.append(anim_motion) # when fps is 30
                else:
                    anim_motion = read_motion(motion_path, fps=args.fps)
                    if motion_name == 'walking' or motion_name == 'running':
                        if anim_motion[1] < min_frame_num :
                            min_frame_num = anim_motion[1]
                    motion_list.append(anim_motion)

            for origin in motion_info["origins"]:
                origin = tuple([origin[0] + offsetX, origin[1] + offsetY, origin[2]])
                origin_list.append(origin)
            for _rotation in motion_info["rotation"]:
                rot_list.append(tuple(_rotation))

        assert len(motion_list) == len(origin_list) == len(rot_list) == len(human_list)


    print('min_frame_num:', min_frame_num) # 1396
    with xf_runner.Sequence.new(seq_name=args.sequence_name, seq_length=min_frame_num, replace=True) as seq:
        actor_list = []
        for i, (motion_data, actor_path, stentcil_val) in enumerate(zip(motion_list, human_list, stencil_list)):   
            actor_list.append(xf_runner.Actor.import_from_file(file_path=actor_path, stencil_value=stentcil_val))
            apply_scale(actor_list[-1].name, scale_factor=args.actor_scale_factor)  # SMPL-XL model is imported with scale, we need to apply scale to it
            actor_list[-1].location = origin_list[i]
            actor_list[-1].rotation = rot_list[i]
            xf_runner.utils.apply_motion_data_to_actor(motion_data=motion_data[0], actor_name=actor_list[-1].name)

        print('actor_list:', actor_list)
        if args.hdri_path:
                xf_runner.utils.set_hdr_map(hdr_map_path=args.hdri_path)


        for i in range(num_of_cameras_for_orbit):
            azimuth = 360 / num_of_cameras_for_orbit * i
            azimuth_radians = math.radians(azimuth)
            x = radius_to_actor * math.cos(azimuth_radians) + actors_center[0]
            y = radius_to_actor * math.sin(azimuth_radians) + actors_center[1]
            z = altitude + actors_center[2]
            location = (x, y, z)
            print(location, actors_center)
            rotation = xf_runner.utils.get_rotation_to_look_at(location=location, target=actors_center)
            print(rotation)

            tot_Location.append(location)
            tot_Rotation.append(rotation)  

        # Spawn static cameras
        for i, (location, rotation) in enumerate(zip(tot_Location, tot_Rotation)):
            # Add a static camera
            static_camera = seq.spawn_camera(
                camera_name=f'static_camera_{i}',
                location=location,
                rotation=rotation,
                fov=args.fov,
            )
            seq.use_camera(camera=static_camera)
        

        # Add a render job to renderer
        # In render job, you can specify the output path, resolution, render passes, etc.
        # The output path is the path to save the rendered data.
        # The resolution is the resolution of the rendered image.
        # The render passes define what kind of data you want to render, such as img, depth, normal, etc.
        # and what kind of format you want to save, such as png, exr, etc.
        seq.add_to_renderer(
            output_path=args.output_path,
            resolution=(1280, 720),
            render_passes=[RenderPass('img', 'png'),
                        RenderPass('depth', 'exr'),  
                        RenderPass('mask', 'exr')], 
            render_samples=32,  # default value 128
            transparent_background=True, 
        )




    # Save the camera trajectory to a json file 
    R_BlenderView_to_OpenCVView = np.diag([1,-1,-1])
    fl_x=745.38
    image_width, image_height  = 1280, 720
    znear, zfar = 0.01, 100.0
    FoVx, FoVy = focal2fov(fl_x, image_width), focal2fov(fl_x, image_height)
    res_dict = {'R': [], 'T': [], 'fl_x': 745.38}
    for rot_, loc_ in zip(tot_Rotation, tot_Location):
        R_BlenderView = R.from_euler('xyz', rot_, degrees=True).as_matrix()
        T_BlenderView = np.array(loc_)
        R_OpenCV = R_BlenderView_to_OpenCVView @ np.transpose(R_BlenderView)
        T_OpenCV = -1.0 * R_OpenCV @ T_BlenderView
        R_OpenCV = np.transpose(R_OpenCV)
        res_dict['R'].append([[element for element in row] for row in R_OpenCV])
        res_dict['T'].append([row for row in T_OpenCV])

    cam_file_path = os.path.join(args.output_path, args.sequence_name, 'camera_trajectory.json')
    with open(cam_file_path, "w") as outfile:
        json.dump(res_dict, outfile)
    print(f'Results saved to "{cam_file_path}".')

    # Save sten: label
    lbl_stencil_dict ={}
    for lbl, sten in zip(lbl_list, stencil_list):
        lbl_stencil_dict[sten] = lbl
    with open(os.path.join(args.output_path, args.sequence_name, 'lbl_stencil.json'), 'w') as f:
        json.dump(lbl_stencil_dict, f)

    # Render
    xf_runner.render()




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--engine_exec_path', type=str, 
                        default='/mnt/hdd/code/blender/blender-3.6.9-linux-x64/blender', 
                        help='Path to the blender executable')
    parser.add_argument('--background', default=False, action='store_true', help='Run blender in background')
    parser.add_argument('--background_mesh_file', type=str, default=None, help='Background mesh file')
    parser.add_argument('--actor_template_path', type=str, default=None, help='actor template path')
    parser.add_argument('--actor_info_file', type=str, default=None, help='actor info file')
    parser.add_argument('--hdri_path', type=str, default=None, help='hdri path')
    parser.add_argument('--output_path', type=str, default=None, help='output path')
    parser.add_argument('--num_actors', type=int, default=12, help='number of actors')
    parser.add_argument('--actor_scale_factor', type=float, default=0.1, help='actor scale factor')
    parser.add_argument('--altitude', type=float, default=1.0, help='altitude')
    parser.add_argument('--sequence_name', type=str, default='test', help='sequence name')
    parser.add_argument('--use_plane', default=False, action='store_true', help='Use plane')
    parser.add_argument('--fov', type=int, default=90, help='Field of view')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    args = parser.parse_args()

    args.background_mesh_file = '/mnt/hdd/data/Okutama_Action/Yonghan_data/okutama_n50_Noon/PoissonMeshes_aligned/fused_sor_lod8.ply'
    args.actor_template_path = '/mnt/hdd/data/SynBody/SMPL-XL-1000-fbx'
    args.actor_info_file = './data/okutama_actor.json'

    main(args)
