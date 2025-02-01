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
    OUT_FOLDER = os.path.join(args.output_path, args.sequence_name)
    os.makedirs(OUT_FOLDER, exist_ok=True)

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
        plane_size = 30
        xf_runner.Shape.spawn_plane(name='plane', location=(0,0,-3.0))
        generate_plane('plane', size=plane_size)

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
                origin = tuple([origin[0] + args.offsetX, origin[1] + args.offsetY, origin[2]])
                origin_list.append(origin)
            for _rotation in motion_info["rotation"]:
                rot_list.append(tuple(_rotation))

        assert len(motion_list) == len(origin_list) == len(rot_list) == len(human_list)

    ###################
    # Set Waypoints
    ###################
    minimum_zvalue = - 3.0
    altitude = args.altitude + minimum_zvalue

    Locations = [
        (-0.27656, 0.20592, altitude), 
        (-1.2, 0.20592, altitude),
        (-0.27656, 1.20592, altitude), 
        (-1.2, 1.20592, altitude),
    ]
    Rotations = [
        (-0.4816, 1.4185, 177.02),
        (-0.4816, 1.4185, 177.02),
        (-0.4816, 1.4185, 177.02),
        (-0.4816, 1.4185, 177.02),
    ]

    for idx in range(len(Locations)):
        Locations[idx] = tuple([Locations[idx][0] + args.offsetX, Locations[idx][1] + args.offsetY, Locations[idx][2]])
    assert len(Locations) == len(Rotations)

    tot_Rotation, tot_Location = [], []
    ts = np.linspace(0, 1, args.interpolate_steps)
    for idx in range(len(Locations) - 1):
        cur_location, nex_location = Locations[idx], Locations[idx + 1]
        cur_rotation, nex_rotation = Rotations[idx], Rotations[idx + 1]
        rots = [cur_rotation for _ in ts]
        trans =  [tuple((1 - t) * np.array(cur_location) + t * np.array(nex_location)) for t in ts]
        tot_Location.extend(trans)
        tot_Rotation.extend(rots)  

    ###################
    # Set Camera orbit
    ###################
    USE_CAMERA_ORBIT = True
    actors_center = (-0.5, 0.5, -3.0)
    radius_to_actor = 1.0
    num_of_cameras_for_orbit = 10
    actors_center = tuple([actors_center[0] + args.offsetX, actors_center[1] + args.offsetY, actors_center[2]])

    # assert Locations[0][-1] == (actors_center[-1] + altitude)

    print('min_frame_num:', min_frame_num)
    #  Start xf_runner
    with xf_runner.Sequence.new(seq_name=args.sequence_name, seq_length=min_frame_num, replace=True) as seq:
        actor_list = []
        for i, (motion_data, actor_path, stentcil_val) in enumerate(zip(motion_list, human_list, stencil_list)):   
            actor_list.append(xf_runner.Actor.import_from_file(file_path=actor_path, stencil_value=stentcil_val))
            apply_scale(actor_list[-1].name, scale_factor=args.actor_scale_factor)  # SMPL-XL model is imported with scale, we need to apply scale to it
            actor_list[-1].location = origin_list[i]
            actor_list[-1].rotation = rot_list[i]
            xf_runner.utils.apply_motion_data_to_actor(motion_data=motion_data[0], actor_name=actor_list[-1].name)
        pdb.set_trace()
        ###################
        # Set Camera orbit
        ###################
        if USE_CAMERA_ORBIT:
            # tot_Rotation, tot_Location = [], []
            for i in range(num_of_cameras_for_orbit):
                azimuth = 360 / num_of_cameras_for_orbit * i
                azimuth_radians = math.radians(azimuth)
                x = radius_to_actor * math.cos(azimuth_radians) + actors_center[0]
                y = radius_to_actor * math.sin(azimuth_radians) + actors_center[1]
                z = args.altitude + actors_center[2]
                location = (x, y, z)
                rotation = xf_runner.utils.get_rotation_to_look_at(location=location, target=actors_center)

                # from temp import look_at
                # render_c2ws = look_at(np.array(location), np.array(actors_center), up=np.array([0., 0., 1.]))
                # rotation = R.from_matrix(render_c2ws[:3,:3]).as_euler('xyz', degrees=True) #
                # rotation = tuple(rotation)
                # print(rotation)

                # rot_mat  = get_rotation_matrix(location, actors_center)
                # rotation = R.from_matrix(rot_mat).as_euler('xyz', degrees=True) # this rotation is different from the above rotation
                # rotation = tuple(rotation)

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
            render_samples=32,  # default value 128
            transparent_background=True, 
        )
        # RenderPass('img', 'png'),

        # export verts of meshes in this sequence and its level
        # xf_runner.utils.export_vertices(export_path=output_path / seq_2_name / 'vertices')

    # Save the camera trajectory to a json file 
    R_BlenderView_to_OpenCVView = np.diag([1,-1,-1])
    fl_x=745.38
    image_width, image_height  = args.resolution
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

    cam_file_path = os.path.join(OUT_FOLDER, 'camera_trajectory.json')
    with open(cam_file_path, "w") as outfile:
        json.dump(res_dict, outfile)
    print(f'Results saved to "{cam_file_path}".')

    # Save sten: label
    lbl_stencil_dict ={}
    for lbl, sten in zip(lbl_list, stencil_list):
        lbl_stencil_dict[sten] = lbl
    with open(os.path.join(OUT_FOLDER, 'lbl_stencil.json'), 'w') as f:
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
    parser.add_argument('--offsetX', type=float, default=5, help='Offset for controlling waypoints, camera orbit, and actor origins')
    parser.add_argument('--offsetY', type=float, default=3, help='Offset for controlling waypoints, camera orbit, and actor origins')
    parser.add_argument('--interpolate_steps', type=int, default=4, help='interpolation steps')
    parser.add_argument('--sequence_name', type=str, default='test', help='sequence name')
    parser.add_argument('--use_plane', default=False, action='store_true', help='Use plane')
    parser.add_argument('--fov', type=int, default=90, help='Field of view')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--resolution', type=int, nargs='+', default=[1280, 720], help='resolution')
    args = parser.parse_args()

    main(args)


