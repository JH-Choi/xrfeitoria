import json
import math
import numpy as np 
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from xrfeitoria.utils.colmap_utils import read_model, convert_to_blender_coord, qvec2rotmat
from xrfeitoria.utils.camera_utils import quaternion_slerp, focal2fov, get_rotation_matrix

import xrfeitoria as xf
from xrfeitoria.data_structure.models import SequenceTransformKey as SeqTransKey
import pdb

# Replace with your executable path
engine_exec_path = '/mnt/hdd/code/blender/blender-3.6.9-linux-x64/blender'

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

# Load Colmap data | Noon 497 images | Morning 715 images
root_path = Path('/mnt/hdd/data/Okutama_Action/Yonghan_data/okutama_n50_Noon')
# root_path = Path('/mnt/hdd/data/Okutama_Action/Yonghan_data/okutama_n50_Morning')
background_mesh_file = root_path / 'PoissonMeshes_aligned' / 'fused_sor_lod8.ply'
fov = 90

# Load Background Current mesh 
xf_runner.utils.import_file(file_path=background_mesh_file)
print('Load background mesh')

# ##########################
# # Set Waypoints
# ##########################
# Locations = [
#     (0.27656, 0.20592, 0.0), 
#     (1.2, 0.20592, 0.0),
#     (0.27656, 1.20592, 0.0), 
#     (1.2, 1.20592, 0.0),
# ]
# Rotations = [
#     (-0.4816, 1.4185, 177.02),
#     (-0.4816, 1.4185, 177.02),
#     (-0.4816, 1.4185, 177.02),
#     (-0.4816, 1.4185, 177.02),
# ]
# steps = 10
# assert len(Locations) == len(Rotations)

# tot_Rotation, tot_Location = [], []
# ts = np.linspace(0, 1, steps)
# for idx in range(len(Locations) - 1):
#     cur_location = Locations[idx]
#     nex_location = Locations[idx + 1]
#     cur_rotation = Rotations[idx]
#     nex_rotation = Rotations[idx + 1]   
#     rots = [cur_rotation for _ in ts]
#     trans =  [tuple((1 - t) * np.array(cur_location) + t * np.array(nex_location)) for t in ts]
#     tot_Location.extend(trans)
#     tot_Rotation.extend(rots)  

###################
# Set Camera orbit
###################
actors_center = (0.5, 0.5, -1.5)
altitude = 1.5 
radius_to_actor = 1.0
num_of_cameras = 10
tot_Rotation, tot_Location = [], []
for i in range(num_of_cameras):
    azimuth = 360 / num_of_cameras * i
    azimuth_radians = math.radians(azimuth)
    x = radius_to_actor * math.cos(azimuth_radians) + actors_center[0]
    y = radius_to_actor * math.sin(azimuth_radians) + actors_center[1]
    z = altitude + actors_center[2]
    location = (x, y, z)
    rotation = xf_runner.utils.get_rotation_to_look_at(location=location, target=actors_center)
    tot_Location.append(location)
    tot_Rotation.append(rotation)  


# Save the camera trajectory to a json file 
data_file_path = None
# data_file_path = 'camera_trajectory.json'

if data_file_path is not None:
    R_BlenderView_to_OpenCVView = np.diag([1,-1,-1])
    fl_x=745.38
    image_width, image_height  = 1280, 720
    znear, zfar = 0.01, 100.0
    FoVx, FoVy = focal2fov(fl_x, image_width), focal2fov(fl_x, image_height)
    res_dict = {'R': [], 'T': [], 'fl_x': 745.38}
    for rot_, loc_ in zip(tot_Rotation, tot_Location):
        R_BlenderView = R.from_euler('xyz', rot_, degrees=True).as_matrix()
        T_BlenderView = np.array(loc_)
        T_BlenderView = -1.0 * R_BlenderView @ loc_ 
        R_OpenCV = R_BlenderView_to_OpenCVView @ np.transpose(R_BlenderView)
        T_OpenCV = -1.0 * R_OpenCV @ T_BlenderView
        R_OpenCV = np.transpose(R_OpenCV)
        res_dict['R'].append([[element for element in row] for row in R_OpenCV])
        res_dict['T'].append([row for row in T_OpenCV])

    with open(data_file_path, "w") as outfile:
        json.dump(res_dict, outfile)
    print(f'Results saved to "{data_file_path}".')

print("the number of transform keys: ", len(tot_Location))
sequence_name = 'MySequence'
frame_num = len(tot_Location)

with xf_runner.Sequence.new(seq_name=sequence_name, seq_length=frame_num, replace=True) as seq:
    for idx, (location, rotation) in enumerate(zip(tot_Location, tot_Rotation)):
        # Add a static camera
        static_camera = seq.spawn_camera(
            camera_name=f'static_camera_{idx}',
            location=location,
            rotation=rotation,
            fov=fov,
        )

        # use the `camera` in level to render
        seq.use_camera(camera=static_camera)
    
    pdb.set_trace()