import math
import numpy as np 
import xrfeitoria as xf
from pathlib import Path
from xrfeitoria.data_structure.models import RenderPass
from xrfeitoria.data_structure.models import SequenceTransformKey as SeqTransKey
from xrfeitoria.rpc import remote_blender
from xrfeitoria.utils.anim import load_amass_motion

from scipy.spatial.transform import Rotation as R
from xrfeitoria.utils.colmap_utils import read_model, convert_to_blender_coord, qvec2rotmat
from xrfeitoria.utils.camera_utils import quaternion_slerp


import pdb

# Replace with your executable path
engine_exec_path = '/mnt/hdd/code/blender/blender-3.6.9-linux-x64/blender'

@remote_blender()
def apply_scale(actor_name: str, scale_factor: float):
    import bpy

    obj = bpy.data.objects.get(actor_name)
    # bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    # bpy.data.objects[actor_name].select_set(True)
    bpy.ops.object.transform_apply(scale=True)
    obj.scale.x *= scale_factor
    obj.scale.y *= scale_factor
    obj.scale.z *= scale_factor


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


# Load COLMAP pose
root_path = Path('/mnt/hdd/data/Okutama_Action/Yonghan_data/okutama_n50_Noon')
colmap_path = root_path / 'colmap_aligned'
# root_path = Path('/mnt/hdd/data/Okutama_Action/GS_data/Scenario2/undistorted/')
# colmap_path = root_path / 'sparse/0'
cameras, images, points3D = read_model(str(colmap_path), ext='.bin')
colmap_data = {}
colmap_data['cameras'] = cameras
colmap_data['images'] = images
colmap_data['points3D'] = points3D

intrinsic_param = np.array([camera.params for camera in colmap_data['cameras'].values()])
intrinsic_matrix = np.array([[intrinsic_param[0][0], 0, intrinsic_param[0][2]],
                                [0, intrinsic_param[0][1], intrinsic_param[0][3]],
                                [0, 0, 1]])  # TODO: only supports single camera for now

image_width = np.array([camera.width for camera in colmap_data['cameras'].values()])
image_height = np.array([camera.height for camera in colmap_data['cameras'].values()])
image_quaternion = np.stack([img.qvec for img in colmap_data['images'].values()])
image_translation = np.stack([img.tvec for img in colmap_data['images'].values()])
camera_id = np.stack([img.camera_id for img in colmap_data['images'].values()]) - 1  # make it zero-indexed
image_names = np.stack([img.name for img in colmap_data['images'].values()])
num_image = image_names.shape[0]
sort_image_id = np.argsort(image_names)

max_step = 1
diff_image_translation = image_translation[1:] - image_translation[:-1]
length_diff = np.linalg.norm(diff_image_translation, axis=1)
min_length = np.min(length_diff)
step_list = length_diff / min_length
step_list = np.round(step_list, 0).astype(int)
step_list = np.minimum(step_list, max_step)

transform_keys = []
tot_quats = []
tot_trans = []
for idx in range(sort_image_id.shape[0] - 1):
    ts = np.linspace(0, 1, step_list[idx])
    # ts = np.linspace(0, 1, 10)

    curr_idx = sort_image_id[idx]
    next_idx = sort_image_id[idx + 1]

    tvec_cur = image_translation[curr_idx]
    qvec_cur = image_quaternion[curr_idx]

    tvec_next = image_translation[next_idx]
    qvec_next = image_quaternion[next_idx]

    quats = [quaternion_slerp(qvec_cur, qvec_next, t) for t in ts]
    trans =  [(1 - t) * tvec_cur + t * tvec_next for t in ts]

    tot_quats.extend(quats)
    tot_trans.extend(trans)

for idx, (qvec_w2c, tvec_w2c) in enumerate(zip(tot_quats, tot_trans)):
    tvec, qvec = convert_to_blender_coord(tvec_w2c, qvec_w2c)
    location = (tvec[0], tvec[1], tvec[2])
    # qvec in the order [w,x,y,z]
    qvec = np.array([qvec[1], qvec[2], qvec[3], qvec[0]])
    Rot = R.from_quat(qvec)
    rotation = Rot.as_euler('xyz', degrees=True) 
    rotation = tuple(r for r in rotation) # (156.53, 57.99, 62.94)

    # Add a transform key to the moving camera
    transform_keys.append(
        SeqTransKey(
            frame=idx,
            location=location,
            rotation=rotation,
            interpolation='AUTO',
        )
    )  
 

# Import the skeletal mesh
actor_tmplate ='/mnt/hdd/data/SynBody/SMPL-XL-100-fbx/{:07}/SMPL-XL-baked.fbx' 

running_motion_path = '/mnt/hdd/data/AMASS/SMPL-X_N/ACCAD/Male2Running_c3d/C3_-_run_stageii.npz'
running_origin = (0, 0, -1.5)
walking_motion_path = '/mnt/hdd/data/AMASS/SMPL-X_N/ACCAD/Male1Walking_c3d/Walk_B10_-_Walk_turn_left_45_stageii.npz'
walking_origin = (2, 0, -1.5)
lying_motion_path = '/mnt/hdd/data/AMASS/SMPL-X_N/ACCAD/Male1General_c3d/General_A9_-__Lie_Down_stageii.npz'
lying_origin = (1, 1, -1.5)
# lying_motion_path = '/mnt/hdd/data/AMASS/SMPL-X_N/ACCAD/MaleGeneral_c3d/General_A9_-___Lie_(forward)_stageii.npz'
sitting_motion_path = '/mnt/hdd/data/AMASS/SMPL-X_N/BMLrub/rub001/0009_sitting1_stageii.npz'
sitting_origin = (0.2, 3.1, -1.5)
standing_motion_path = '/mnt/hdd/data/AMASS/SMPL-X_N/ACCAD/Male2General_c3d/A1-_Stand_stageii.npz'
standing_origin = (2.3, 2.6, -1.5)
# standing_motion_path = '/mnt/hdd/data/AMASS/SMPL-X_N/ACCAD/Female1General_c3d/A1-_Stand_stageii.npz'
fbx_list = [1,2,6,7,10]
stencil_list = [50, 100, 150, 200 ,250]
assert len(fbx_list) == len(stencil_list)

camera_origin = (1.45, 1.068, -1.5)

motion_list = []
motion_list.append(read_motion(running_motion_path))
motion_list.append(read_motion(walking_motion_path))
motion_list.append(read_motion(lying_motion_path))
motion_list.append(read_motion(sitting_motion_path))
motion_list.append(read_motion(standing_motion_path))

origin_list = []
origin_list.append(running_origin)
origin_list.append(walking_origin)  
origin_list.append(lying_origin)
origin_list.append(sitting_origin)
origin_list.append(standing_origin)

assert len(motion_list) == len(origin_list)

max_frame_num = 0
for _, n_frames in motion_list:  
    if n_frames > max_frame_num:
        max_frame_num = n_frames

print('max_frame_num:', max_frame_num) # 1396

# Hyperparameters
# center = (411.03, -2.6848, -40.564)  # Mega nerf dataset
center = (2.67, 1.676, -1.4)  # Okuama dataset
altitude = 5 
radius_to_actor = 0.5
num_of_cameras = 6
# Set cameras' field of view to 90°
camera_fov = 90
actor_scale_factor = 0.1

# save the level
if render_engine == 'unreal':
    xf_runner.utils.save_current_level()

# Load Background Current mesh 
# background_mesh_file = Path('/mnt/hdd/data/mega_nerf_data/Mill19/building/building-pixsfm/colmap_aligned/7/mesh_deci0.75.ply')
background_mesh_file = Path('/mnt/hdd/data/Okutama_Action/Yonghan_data/okutama_n50_Noon/PoissonMeshes_aligned/fused_sor_lod8.ply')
xf_runner.utils.import_file(file_path=background_mesh_file)
print('Load background mesh')

# Use `with` statement to create a sequence, and it will be automatically close the sequence after the code block is executed.
# The argument `seq_length` controls the number of frames to be rendered. 
# sequence_name = 'MySequence'
sequence_name = f'alti{altitude}_radi{radius_to_actor}_cam{num_of_cameras}'
with xf_runner.sequence(seq_name=sequence_name, seq_length=max_frame_num) as seq:
    actor_list = []
    for i, (motion_data, fbx_idx, stentcil_val) in enumerate(zip(motion_list, fbx_list, stencil_list)):   
        actor_path = actor_tmplate.format(fbx_idx)
        actor_list.append(xf_runner.Actor.import_from_file(file_path=actor_path, stencil_value=stentcil_val))
        apply_scale(actor_list[-1].name, scale_factor=actor_scale_factor)  # SMPL-XL model is imported with scale, we need to apply scale to it
        actor_list[-1].location = origin_list[i]
        xf_runner.utils.apply_motion_data_to_actor(motion_data=motion_data[0], actor_name=actor_list[-1].name)

    # Modify the frame range to the length of the motion
    frame_start, frame_end = xf_runner.utils.get_keys_range()
    print('frame_start:', frame_start)
    print('frame_end:', frame_end)
    xf_runner.utils.set_frame_range(frame_start, frame_end)
        
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
        output_path=f'./output/{render_engine}_okutama/',
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
# # xf_runner.close()


