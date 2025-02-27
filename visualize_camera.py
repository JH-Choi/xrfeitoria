import json
import math
import numpy as np
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
from xrfeitoria.utils.motion_utils import apply_scale
import xrfeitoria as xf
import pdb

# Function to add Gaussian noise to a vector
def gen_noise(shape=3, noise_std=0.01):
    """
    Generate Gaussian noise to a 3D vector.
    Args:
    - vector (np.array): Original 3D vector (rotation or location).
    - noise_std (float): Standard deviation of the Gaussian noise.

    Returns:
    - np.array: Noisy vector.
    """
    noise = np.random.normal(loc=0, scale=noise_std, size=(shape,))
    return noise


parser = ArgumentParser()
parser.add_argument('--engine_exec_path', type=str, 
                    default='/mnt/hdd/code/blender/blender-3.6.9-linux-x64/blender', 
                    help='Path to the blender executable')
parser.add_argument('--fov', type=int, default=90, help='Field of view')
parser.add_argument('--background', default=False, action='store_true', help='Run blender in background')
parser.add_argument('--background_mesh_file', type=str, default=None, help='Background mesh file')
parser.add_argument('--camera_scale_factor', type=float, default=0.3, help='camera visualization size')
args = parser.parse_args()

xf_runner = xf.init_blender(exec_path=args.engine_exec_path, 
                            background=args.background, 
                            new_process=True)


args.background_mesh_file = '/mnt/hdd/code/outdoor_relighting/PGSR/output/okutama_r2_wg_mip/Scenario2/mesh_maxdepth10_vox0.01/tsdf_fusion_post_deci.ply'
# args.background_mesh_file = '/mnt/hdd/code/outdoor_relighting/PGSR/output/okutama_r2_wg_mip/Scenario3/mesh_maxdepth10_vox0.01/tsdf_fusion_post_deci.ply'

#####################################################
# Load camera location from the JSON file
#####################################################
# camera_file = "Drone2_Noon_2_2_2_pose_from_colmap.json"
# with open(camera_file, "r") as f:
#     data = json.load(f)

# tot_Rotation, tot_Location = [], []
# for camera_name in data.keys():
#     cam_pos  = np.array(data[camera_name]["T"])
#     cam_rot  = np.array(data[camera_name]["R"])
#     # cam_rot = np.array([0,0,0])
#     tot_Rotation.append(cam_rot)
#     tot_Location.append(cam_pos)    

#####################################################
# Camera from waypoint (Orbit Trajectory)
#####################################################
# num_of_cameras_for_orbit = 20
# radius_to_actor = 2.0
# altitude = 3.0
# actors_center = (0, 0, -3.0) # basic zcoord of mesh is -3.0

# tot_Rotation, tot_Location = [], []
# for i in range(num_of_cameras_for_orbit):
#     azimuth = 360 / num_of_cameras_for_orbit * i
#     azimuth_radians = math.radians(azimuth)
#     x = radius_to_actor * math.cos(azimuth_radians) + actors_center[0]
#     y = radius_to_actor * math.sin(azimuth_radians) + actors_center[1]
#     z = actors_center[2] + altitude
#     location = (x, y, z)
#     rotation = xf_runner.utils.get_rotation_to_look_at(location=location, target=actors_center)
#     rotation[0] = 55

#     # Add noise
#     rotation = rotation + gen_noise(noise_std=0.05)  # Increase noise level if needed
#     location = location + gen_noise(noise_std=0.1)

#     tot_Location.append(location)
#     tot_Rotation.append(rotation)  


#####################################################
# Stationary Yaw Rotation
# Control Rz
#####################################################
num_of_cameras_for_orbit = 20
fixed_position = (0, 0, -3.0) # basic zcoord of mesh is -3.0
altitude = 3.0
Rx, Ry = 55.0, 0
fixed_position = (fixed_position[0], fixed_position[1], fixed_position[2] + altitude)   

tot_Rotation, tot_Location = [], []
for i in range(num_of_cameras_for_orbit):
    azimuth = 360 / num_of_cameras_for_orbit * i
    location = fixed_position
    rotation = (Rx, Ry, azimuth)

    # Add noise
    rotation = rotation + gen_noise(noise_std=0.05)  # Increase noise level if needed
    location = location + gen_noise(noise_std=0.1)

    tot_Location.append(location)
    tot_Rotation.append(rotation)  



if args.background_mesh_file is not None:
    xf_runner.utils.import_file(file_path=args.background_mesh_file)
    print('Load background mesh')

sequence_name = 'MySequence'
frame_num = len(tot_Location)
with xf_runner.Sequence.new(seq_name=sequence_name, seq_length=frame_num, replace=True) as seq:
    for idx, (location, rotation) in enumerate(zip(tot_Location, tot_Rotation)):
        # Add a static camera
        static_camera = seq.spawn_camera(
            camera_name=f'static_camera_{idx}',
            location=location,
            rotation=rotation,
            fov=args.fov,
        )

        apply_scale(f'static_camera_{idx}', scale_factor=args.camera_scale_factor) 

        # use the `camera` in level to render
        seq.use_camera(camera=static_camera)
    pdb.set_trace()