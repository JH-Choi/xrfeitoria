import os
import math
import json
import random
import numpy as np 
import xrfeitoria as xf
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

# Replace with your executable path
engine_exec_path = '/mnt/hdd/code/blender/blender-3.6.9-linux-x64/blender'


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
def load_asset_name(text_name):
    out = []
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            out.append(obj.name)
    with open(text_name, 'w') as text_file:
        for name in out:
            text_file.write(name + '\n')  
    text_file.close()

# def read_asset_names(text_name):    
#     with open(text_name, 'r') as text_file:
#         mesh_names = [line.strip() for line in text_file.readlines()]
#     return mesh_names


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

exec_path_stem = Path(engine_exec_path).stem.lower()
if 'blender' in exec_path_stem:
    # Open Blender
    render_engine = 'blender'
    xf_runner = xf.init_blender(exec_path=engine_exec_path, 
                                background=False, # False 
                                new_process=True)
elif 'unreal' in exec_path_stem:
    # Unreal Engine requires a project to be opened
    # Here we use a sample project, which is downloaded from the following link
    # You can also use your own project
    import shutil
    from xrfeitoria.utils.downloader import download
    unreal_project_zip = download(url='https://openxrlab-share.oss-cn-hongkong.aliyuncs.com/xrfeitoria/tutorials/unreal_project/UE_Sample.zip', 
                                    dst_dir="./tutorial03/assets/")
    shutil.unpack_archive(filename=unreal_project_zip, extract_dir='./tutorial03/assets/')

    # Open Unreal Engine
    render_engine = 'unreal'
    xf_runner = xf.init_unreal(exec_path=engine_exec_path, 
                                background=False, 
                                new_process=True, 
                                project_path='./tutorial03/assets/UE_sample/UE_sample.uproject')

# Set Hyperparameters
# Load Colmap data | Noon 497 images | Morning 715 images
root_path = Path('/mnt/hdd/data/Okutama_Action/Yonghan_data/okutama_n50_Noon')
# root_path = Path('/mnt/hdd/data/Okutama_Action/Yonghan_data/okutama_n50_Morning')
# hdri_path = './dry_orchard_meadow_4k.exr'
hdri_path = None
background_mesh_file = root_path / 'PoissonMeshes_aligned' / 'fused_sor_lod8.ply'
assert background_mesh_file.exists()
fov = 90
num_actors = 12
actor_scale_factor = 0.1
fps = 30 # 30, 120, None
offsetX = 5  # Offset for controlling waypoints, camera orbit, and actor origins
offsetY = 3  # Offset for controlling waypoints, camera orbit, and actor origins
steps = 4
altitude = 1.0
output_path = f'./output/{render_engine}_waypoint/'
sequence_name = f'altitude{altitude}_offsetX{offsetX}_offsetY{offsetY}'

# Load Actors
actor_template_path = Path('/mnt/hdd/data/SynBody/SMPL-XL-1000-fbx')
actor_list = [d for d in actor_template_path.iterdir() if d.is_dir()]
fbx_list = random.sample(actor_list, num_actors)
fbx_list = [str(fbx / 'SMPL-XL-baked.fbx') for fbx in fbx_list]

# for fbx_file in fbx_list:
#     print(fbx_file)
#     assert Path(fbx_file).exists()
#     asset_name = load_fbx_object_names(fbx_file)
#     print(asset_name)
#     pdb.set_trace() 



@remote_blender()
def generate_plane(actor_name: str, size):
    plane = bpy.data.objects.get(actor_name)
    plane.select_set(True)
    plane.scale = (size, size, 1)
    plane.is_shadow_catcher = True


###################
# Load Plane for shadow catcher
###################
# size = 30
# xf_runner.Shape.spawn_plane(name='plane', location=(0,0,-1.5))
# generate_plane('plane', size=size)


###################
# # Load Background Current mesh 
###################
# xf_runner.utils.import_file(file_path=background_mesh_file)
# print('Load background mesh')


# Load the motion data from the JSON file
with open('./data/okutama_actor.json', 'r') as json_file:
    motion_dict = json.load(json_file)

stencil_list = [int(stencil) for stencil in np.linspace(10, 255, num_actors)]
assert len(fbx_list) == len(stencil_list)

###################
# Set Waypoints
###################
Locations = [
    (0.27656, 0.20592, altitude), 
    (1.2, 0.20592, altitude),
    (0.27656, 1.20592, altitude), 
    (1.2, 1.20592, altitude),
]
Rotations = [
    (-0.4816, 1.4185, 177.02),
    (-0.4816, 1.4185, 177.02),
    (-0.4816, 1.4185, 177.02),
    (-0.4816, 1.4185, 177.02),
]

for idx in range(len(Locations)):
    Locations[idx] = tuple([Locations[idx][0] + offsetX, Locations[idx][1] + offsetY, Locations[idx][2]])
assert len(Locations) == len(Rotations)

tot_Rotation, tot_Location = [], []
ts = np.linspace(0, 1, steps)
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
USE_CAMERA_ORBIT = False
actors_center = (0.5, 0.5, -1.5)
altitude = altitude + 1.5 
radius_to_actor = 1.0
num_of_cameras_for_orbit = 10
actors_center = tuple([actors_center[0] + offsetX, actors_center[1] + offsetY, actors_center[2]])

assert Locations[0][-1] == (actors_center[-1] + altitude)


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
            anim_motion = read_motion(motion_path, fps=fps, start_frame=78, end_frame=135)
            motion_list.append(anim_motion) # when fps is 30
        else:
            anim_motion = read_motion(motion_path, fps=fps)
            if motion_name == 'walking' or motion_name == 'running':
                if anim_motion[1] < min_frame_num :
                    min_frame_num = anim_motion[1]
            motion_list.append(anim_motion)
    for origin in motion_info["origins"]:
        origin = tuple([origin[0] + offsetX, origin[1] + offsetY, origin[2]])
        origin_list.append(origin)
    for _rotation in motion_info["rotation"]:
        rot_list.append(tuple(_rotation))

assert len(motion_list) == len(origin_list) == len(rot_list) == len(fbx_list)

print('min_frame_num:', min_frame_num) # 1396

os.makedirs(os.path.join(output_path, sequence_name), exist_ok=True)

all_assets_name = []    
actor_info_dict = {}
text_path = 'actor_info.txt'

min_frame_num=10
#  Start xf_runner
with xf_runner.Sequence.new(seq_name=sequence_name, seq_length=min_frame_num, replace=True) as seq:
    actor_list = []
    for i, (motion_data, actor_path, stentcil_val) in enumerate(zip(motion_list, fbx_list, stencil_list)):   
        actor_list.append(xf_runner.Actor.import_from_file(file_path=actor_path, stencil_value=stentcil_val))
        # actor_name = actor_list[-1].name 
        load_asset_name(text_path)
        # import pdb; pdb.set_trace()
        # out = read_asset_names(text_path)

        # actor_info_dict[actor_name] = []
        # for asset_name in out:
        #     if asset_name not in all_assets_name:
        #         actor_info_dict[actor_name].append(asset_name)
        # all_assets_name.extend(out)

        # out = load_asset_name(actor_list[-1].name)
        # actor_info_dict[actor_list[-1].name] = out  
        apply_scale(actor_list[-1].name, scale_factor=actor_scale_factor)  # SMPL-XL model is imported with scale, we need to apply scale to it
        actor_list[-1].location = origin_list[i]
        actor_list[-1].rotation = rot_list[i]
        xf_runner.utils.apply_motion_data_to_actor(motion_data=motion_data[0], actor_name=actor_list[-1].name)

    ###################
    # Set Camera orbit
    # If I run this part before load actor, it causes error but I don't know why
    ###################
    if USE_CAMERA_ORBIT:
        # tot_Rotation, tot_Location = [], []
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

    #############################################
    ### Spawn static cameras
    #############################################
    # for i, (location, rotation) in enumerate(zip(tot_Location, tot_Rotation)):
    #     # Add a static camera
    #     static_camera = seq.spawn_camera(
    #         camera_name=f'static_camera_{i}',
    #         location=location,
    #         rotation=rotation,
    #         fov=fov,
    #     )
    #     seq.use_camera(camera=static_camera)
    
    if hdri_path:
        xf_runner.utils.set_hdr_map(hdr_map_path=hdri_path)

    # Add a render job to renderer
    # In render job, you can specify the output path, resolution, render passes, etc.
    # The output path is the path to save the rendered data.
    # The resolution is the resolution of the rendered image.
    # The render passes define what kind of data you want to render, such as img, depth, normal, etc.
    # and what kind of format you want to save, such as png, exr, etc.
    # seq.add_to_renderer(
    #     output_path=output_path,
    #     resolution=(1280, 720),
    #     render_passes=[RenderPass('img', 'png'),
    #                    RenderPass('depth', 'exr'),  
    #                    RenderPass('mask', 'exr')], 
    #     render_samples=32,  # default value 128
    #     transparent_background=True, 
    # )
    
    # export verts of meshes in this sequence and its level
    export_path = os.path.join(output_path, sequence_name, 'vertices')
    xf_runner.utils.export_vertices(export_path=export_path, use_animation=True)
    # xf_runner.utils.export_vertices(export_path=export_path)
    # xf_runner.utils.create_render_package(
    #     export_path=export_path, 
    #     use_animation=True
    # )

actor_names = [actor.name for actor in actor_list]
print(actor_names)

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

cam_file_path = os.path.join(output_path, sequence_name, 'camera_trajectory.json')
with open(cam_file_path, "w") as outfile:
    json.dump(res_dict, outfile)
print(f'Results saved to "{cam_file_path}".')

# Save sten: label
lbl_stencil_dict ={}
for lbl, sten in zip(lbl_list, stencil_list):
    lbl_stencil_dict[sten] = lbl
with open(os.path.join(output_path, sequence_name, 'lbl_stencil.json'), 'w') as f:
    json.dump(lbl_stencil_dict, f)

# Render
# xf_runner.render()

