import math
import xrfeitoria as xf
from pathlib import Path
from xrfeitoria.data_structure.models import RenderPass
from xrfeitoria.data_structure.models import SequenceTransformKey as SeqTransKey
from xrfeitoria.rpc import remote_blender
from xrfeitoria.utils import setup_logger

import numpy as np 
from xrfeitoria.utils.colmap_utils import read_model
from xrfeitoria.utils.anim import dump_humandata, load_amass_motion
import pdb

# Replace with your executable path
# engine_exec_path = 'C:/Program Files/Blender Foundation/Blender 3.6/blender.exe'
# engine_exec_path = '/mnt/hdd/code/human_data_generation/DroneSim/dronesim/foreground/Blender/blender-3.3.6-linux-x64/blender'
# engine_exec_path = '/mnt/hdd/code/human_data_generation/DroneSim/dronesim/foreground/Blender/blender-3.3.6-linux-x64/blender'
# engine_exec_path = '/mnt/hdd/code/blender/blender-3.3.6-stable+v33.948f8298b982-linux.x86_64-release/blender'
engine_exec_path = '/mnt/hdd/code/blender/blender-3.6.9-linux-x64/blender'
# engine_exec_path = 'C:/Program Files/Epic Games/UE_5.1/Engine/Binaries/Win64/UnrealEditor-Cmd.exe'

# @remote_blender()
# def apply_scale(actor_name: str):
#     import bpy

#     bpy.ops.object.select_all(action='DESELECT')
#     bpy.data.objects[actor_name].select_set(True)
#     bpy.ops.object.transform_apply(scale=True)

@remote_blender()
def apply_scale(actor_name: str):
    import bpy

    obj = bpy.data.objects.get(actor_name)
    # bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    # bpy.data.objects[actor_name].select_set(True)
    bpy.ops.object.transform_apply(scale=True)
    obj.scale.x *= 0.2
    obj.scale.y *= 0.2
    obj.scale.z *= 0.2


################################################################################################################################
# 03_basic_render.py in blender folder
# In XRFeitoria, a `level` is an editable space that can be used to place objects(3D models, lights, cameras, etc.),
# and a `sequence` is a collection of objects in `level` that can be rendered together.

# The objects in a `level` are shared by all `sequence`s in this `level`,
# and the objects in a `sequence` are independent of other `sequence`s in this `level`.
# Therefore, rendering should be performed on a `sequence` basis.

# In blender, a `level` is a `scene`, and a `sequence` is a `collection` in the `scene`.
# The default level is named `XRFeitoria`, and it is automatically created when the blender is started.
# The operations in the previous examples are all performed in the default level.
# You can also use other preset levels downloaded from the internet, or create your own level by xf_runner.utils.new_level().

# The following example demonstrates how to use a preset level to render a scene.
################################################################################################################################

exec_path_stem = Path(engine_exec_path).stem.lower()
if 'blender' in exec_path_stem:
    # Open Blender
    render_engine = 'blender'
    xf_runner = xf.init_blender(exec_path=engine_exec_path, 
                                background=False, 
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

# Import the skeletal mesh
actor1_path = '/mnt/hdd/data/SynBody/SMPL-XL-100-fbx/0000001/SMPL-XL-baked.fbx' 
actor2_path = '/mnt/hdd/data/SynBody/SMPL-XL-100-fbx/0000002/SMPL-XL-baked.fbx'
# actor2_motion_path = '/mnt/hdd/data/AMASS/SMPL-X_N/SFU/0005/0005_Walking001_stageii.npz'
actor2_motion_path = '/mnt/hdd/data/AMASS/SMPL-X_N/ACCAD/Male2Running_c3d/C3_-_run_stageii.npz'
# actor1 = xf_runner.Actor.import_from_file(file_path=actor1_path, stencil_value=100)
# actor2 = xf_runner.Actor.import_from_file(file_path=actor2_path, stencil_value=200)
motion = load_amass_motion(actor2_motion_path)  # modify this to motion file in absolute path
motion.convert_fps(30)  # convert the motion from 120fps (amass) to 30fps
# motion.cut_motion(end_frame=10)  # cut the motion to 10 frames, for demonstration purpose
motion_data = motion.get_motion_data()

# center = (381.56, 34.083, -42.94)
# center = (411.03, -2.6848, -40.564)  # Mega nerf dataset
center = (2.67, 1.676, -1.4)  # Okuama dataset
altitude = 5 
radius_to_actor = 5
num_of_cameras = 1
# Set cameras' field of view to 90°
camera_fov = 90
# set actor's size to 0.4m
actor_size = 0.02

# save the level
if render_engine == 'unreal':
    xf_runner.utils.save_current_level()

# # Load Colmap data
# colmap_path = '/mnt/hdd/data/mega_nerf_data/Mill19/building/building-pixsfm/colmap_aligned/7'
# cameras, images, points3D = read_model(colmap_path, ext='.bin')
# colmap_data = {}
# colmap_data['cameras'] = cameras
# colmap_data['images'] = images
# colmap_data['points3D'] = points3D
# intrinsic_param = np.array([camera.params for camera in colmap_data['cameras'].values()])
# intrinsic_matrix = np.array([[intrinsic_param[0][0], 0, intrinsic_param[0][2]],
#                                 [0, intrinsic_param[0][1], intrinsic_param[0][3]],
#                                 [0, 0, 1]])  # TODO: only supports single camera for now

# image_width = np.array([camera.width for camera in colmap_data['cameras'].values()])
# image_height = np.array([camera.height for camera in colmap_data['cameras'].values()])
# image_quaternion = np.stack([img.qvec for img in colmap_data['images'].values()])
# image_translation = np.stack([img.tvec for img in colmap_data['images'].values()])
# camera_id = np.stack([img.camera_id for img in colmap_data['images'].values()]) - 1  # make it zero-indexed
# image_names = np.stack([img.name for img in colmap_data['images'].values()])
# num_image = image_names.shape[0]


# Load Background Current mesh 
# background_mesh_file = Path('/mnt/hdd/data/mega_nerf_data/Mill19/building/building-pixsfm/colmap_aligned/7/mesh_deci0.75.ply')
background_mesh_file = Path('/mnt/hdd/data/Okutama_Action/Yonghan_data/okutama_n50_Noon/mvs/fused_sor_lod8.ply')
xf_runner.utils.import_file(file_path=background_mesh_file)
print('Load background mesh')
print('Center:', center)

# Use `with` statement to create a sequence, and it will be automatically close the sequence after the code block is executed.
# The argument `seq_length` controls the number of frames to be rendered. 
# sequence_name = 'MySequence'
sequence_name = f'alti{altitude}_radi{radius_to_actor}_cam{num_of_cameras}'
frame_num = motion.n_frames
print('frame_num:', frame_num)
# with xf_runner.Sequence.new(seq_name=sequence_name, seq_length=frame_num, replace=True) as seq:
with xf_runner.sequence(seq_name=sequence_name, seq_length=frame_num) as seq:
    actor1 = xf_runner.Actor.import_from_file(file_path=actor1_path, stencil_value=100)
    # actor1 = xf_runner.Actor.import_from_file(file_path=actor1_path, scale=(0.002, 0.002, 0.002), stencil_value=100)
    actor2 = xf_runner.Actor.import_from_file(file_path=actor2_path, stencil_value=100)
    # actor2 = xf_runner.Actor.import_from_file(file_path=actor2_path, scale=(0.002, 0.002, 0.002), stencil_value=200)

    apply_scale(actor1.name)  # SMPL-XL model is imported with scale, we need to apply scale to it
    apply_scale(actor2.name)  # SMPL-XL model is imported with scale, we need to apply scale to it

    actor1_location = center
    actor1.location = actor1_location
    # # Set the location of the two actors to make their distance to be 1.0 meter
    actor2.location = (actor1_location[0], actor1_location[1] + 1.0, actor1_location[2])

    # actor2_bbox = actor2.bound_box
    # actor2_center = ((actor2_bbox[0][0] + actor2_bbox[1][0]) / 2, (actor2_bbox[0][1] + actor2_bbox[1][1]) / 2, (actor2_bbox[0][2] + actor2_bbox[1][2]) / 2)
    # actors_center = ((actor2_center[0] + actor2_center[0]) / 2, (actor2_center[1] + actor2_center[1]) / 2, (actor2_center[2] + actor2_center[2]) / 2)

    print('Apply motion data')
    xf_runner.utils.apply_motion_data_to_actor(motion_data=motion_data, actor_name=actor2.name)
    # xf_runner.utils.apply_motion_data_to_actor(motion_data=motion_data, actor_name=actor2.name, is_first_frame_as_origin=False)
    # set_scale(actor1.name)
    # set_scale(actor2.name)

    # Get the bounding boxes of the actors
    actor1_bbox = actor1.bound_box
    actor2_bbox = actor2.bound_box

    # Get the center location of the actors
    actor1_center = ((actor1_bbox[0][0] + actor1_bbox[1][0]) / 2, (actor1_bbox[0][1] + actor1_bbox[1][1]) / 2, (actor1_bbox[0][2] + actor1_bbox[1][2]) / 2)
    actor2_center = ((actor2_bbox[0][0] + actor2_bbox[1][0]) / 2, (actor2_bbox[0][1] + actor2_bbox[1][1]) / 2, (actor2_bbox[0][2] + actor2_bbox[1][2]) / 2)
    actors_center = ((actor1_center[0] + actor2_center[0]) / 2, (actor1_center[1] + actor2_center[1]) / 2, (actor1_center[2] + actor2_center[2]) / 2)

    print('actors_center:', actors_center)

    # Modify the frame range to the length of the motion
    frame_start, frame_end = xf_runner.utils.get_keys_range()
    print('frame_start:', frame_start)
    print('frame_end:', frame_end)
    xf_runner.utils.set_frame_range(frame_start, frame_end)

    ##########################################################################
    # Add 6 static cameras and a moving camera around the actors for rendering
    ##########################################################################
    # Set cameras' distance to 3.0m
    # distance_to_actor = 3.0
    # Prepare the transform keys for moving camera
    transform_keys = []
    # calculate the location and rotation of the cameras
    for i in range(num_of_cameras):
        azimuth = 360 / num_of_cameras * i
        azimuth_radians = math.radians(azimuth)

        x = radius_to_actor * math.cos(azimuth_radians) + actors_center[0]
        y = radius_to_actor * math.sin(azimuth_radians) + actors_center[1]
        z = altitude + actors_center[2]
        location = (x, y, z)
        # Set camera's rotation to look at the actor's center

        # x = 0.0 + actors_center[0]
        # y = distance_to_actor * math.cos(azimuth_radians) + actors_center[1]
        # z = distance_to_actor * math.sin(azimuth_radians) + actors_center[2]
        # location = (x, y, z)
        # Set camera's rotation to look at the actor's center
 
        rotation = xf_runner.utils.get_rotation_to_look_at(location=location, target=actors_center)
        print(f'camera_{i} location: {location}')
        print(f'camera_{i} rotation: {rotation}')

        # Add a static camera
        static_camera = seq.spawn_camera(
            camera_name=f'static_camera_{i}',
            location=location,
            rotation=rotation,
            fov=camera_fov,
        )

        # use the `camera` in level to render
        seq.use_camera(camera=static_camera)
        
        # Add a transform key to the moving camera
        transform_keys.append(
            SeqTransKey(
                frame=i,
                location=location,
                rotation=rotation,
                interpolation='AUTO',
            )
        )  
    
    # Add a moving camera rotating around the actors
    moving_camera = seq.spawn_camera_with_keys(
        camera_name=f'moving_camera',
        transform_keys=transform_keys,
        fov=camera_fov,
    )

    # Add a render job to renderer
    # In render job, you can specify the output path, resolution, render passes, etc.
    # The output path is the path to save the rendered data.
    # The resolution is the resolution of the rendered image.
    # The render passes define what kind of data you want to render, such as img, depth, normal, etc.
    # and what kind of format you want to save, such as png, exr, etc.
    seq.add_to_renderer(
        output_path=f'./output/{render_engine}/',
        resolution=(1280, 720),
        render_passes=[RenderPass('img', 'png'),
                       RenderPass('mask', 'exr'),
                       RenderPass('normal', 'exr'),
                       RenderPass('diffuse', 'exr')]
    )

    # export verts of meshes in this sequence and its level
    # xf_runner.utils.export_vertices(export_path=output_path / seq_2_name / 'vertices')
    # pdb.set_trace()

# render
xf_runner.render()

# Save the blender file to the current directory
# output_blend_file = output_path / 'source.blend'
# xf_runner.utils.save_blend(save_path=output_blend_file)

# visualize_vertices(
#     camera_name=camera_name,
#     actor_names=actor_names,
#     seq_output_path=output_path / seq_2_name,
#     frame_idx=5,
# )

# Close the blender process
xf_runner.close()


