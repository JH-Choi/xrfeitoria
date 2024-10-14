import math
import xrfeitoria as xf
from pathlib import Path
from xrfeitoria.data_structure.models import RenderPass
from xrfeitoria.data_structure.models import SequenceTransformKey as SeqTransKey
from argparse import ArgumentParser

import numpy as np 
from scipy.spatial.transform import Rotation as R
from xrfeitoria.utils.colmap_utils import read_model, convert_to_blender_coord, qvec2rotmat
import pdb


# def get_random_hemisphere_loc(radius):
#     u = np.random.uniform(0, 1, size=2)
#     z = u[0]
#     r = math.sqrt(max(0, 1. - z * z))
#     phi = 2 * math.pi * u[1]
#     return (radius * r * math.cos(phi), radius * r * math.sin(phi), radius * z)


def get_random_hemisphere_loc(theta_range, z_range):
    """
    Generate a random point on a hemisphere with random z-value and theta (zenith angle) within manual ranges.
    
    Parameters:
    - z_range: A tuple (min_z, max_z) for the range of z-values.
    - theta_range: A tuple (min_theta, max_theta) for the range of theta (in radians).
    
    Returns:
    A tuple (x, y, z) representing the Cartesian coordinates of the point on the hemisphere.
    """
    # Generate a random z-value within the provided range
    z_value = np.random.uniform(z_range[0], z_range[1])

    # Generate a random zenith angle theta within the provided range
    theta = np.random.uniform(theta_range[0], theta_range[1])

    radius = z_value / math.cos(theta)    

    # Check that the z_value is within the hemisphere range
    if z_value < 0 or z_value > radius:
        raise ValueError("The z-value must be within the hemisphere range [0, radius].")
    
    # Compute the radial distance in the xy-plane based on the z-value
    r = math.sqrt(radius**2 - z_value**2)
    
    # Random azimuthal angle phi, constrained to [0, 2*pi]
    phi = np.random.uniform(0, 2 * math.pi)

    # Convert spherical coordinates to Cartesian coordinates
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    z = z_value

    return (x, y, z)


def main(args): 
    ################# Hyperparameters #################
    number_of_cameras = 6
    camera_origin = (-0.7642, 1.8742, -3.0848)
    z_range = (2, 4) # Altitude
    theta_range = (0, math.pi / 6) # Camera pitch angle
    ##################################################

    exec_path_stem = Path(args.engine_exec_path).stem.lower()
    if 'blender' in exec_path_stem:
        # Open Blender
        render_engine = 'blender'
        xf_runner = xf.init_blender(exec_path=args.engine_exec_path, 
                                    background=False, 
                                    new_process=True)

    # Load Colmap data
    # colmap_path = Path(args.colmap_path)

    # cameras, images, points3D = read_model(str(colmap_path), ext='.bin')
    # colmap_data = {}
    # colmap_data['cameras'] = cameras
    # colmap_data['images'] = images
    # colmap_data['points3D'] = points3D


    # # Load colmap data
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
    # sort_image_id = np.argsort(image_names)
    # frame_num = num_image

    # Load Background Current mesh 
    if args.background_mesh_file is not None:
        xf_runner.utils.import_file(file_path=args.background_mesh_file)
        print('Load background mesh')

    sequence_name = 'MySequence'
    frame_num = 10
    with xf_runner.Sequence.new(seq_name=sequence_name, seq_length=frame_num, replace=True) as seq:
        for c_id in range(number_of_cameras):
            pos = get_random_hemisphere_loc(theta_range, z_range)
            pos = tuple([float(p) for p in pos])
            location = tuple([p + camera_origin[i] for i, p in enumerate(pos)])
            # location = camera_origin
            rotation = xf_runner.utils.get_rotation_to_look_at(location=location, target=camera_origin)
            print(f'camera_{c_id} location: {location}')
            print(f'camera_{c_id} rotation: {rotation}')
            # Add a static camera
            static_camera = seq.spawn_camera(
                camera_name=f'static_camera_{c_id}',
                location=location,
                rotation=rotation,
                fov=args.camera_fov,
            )

            # use the `camera` in level to render
            seq.use_camera(camera=static_camera)
    

        # for idx, i_id in enumerate(sort_image_id):
        #     if idx % args.sampling_rate != 0:
        #         continue
        #     tvec_w2c = image_translation[i_id]
        #     qvec_w2c = image_quaternion[i_id]
        #     tvec, qvec = convert_to_blender_coord(tvec_w2c, qvec_w2c)

        #     location = (tvec[0], tvec[1], tvec[2])
        #     # qvec in the order [w,x,y,z]
        #     qvec = np.array([qvec[1], qvec[2], qvec[3], qvec[0]])
        #     Rot = R.from_quat(qvec)
        #     rotation = Rot.as_euler('xyz', degrees=True)
        #     rotation = tuple(r for r in rotation)

        #     # Add a static camera
        #     static_camera = seq.spawn_camera(
        #         camera_name=f'static_camera_{idx}',
        #         location=location,
        #         rotation=rotation,
        #         fov=args.fov,
        #     )

        #     # use the `camera` in level to render
        #     seq.use_camera(camera=static_camera)

        pdb.set_trace()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--engine_exec_path', type=str, 
                        default='/mnt/hdd/code/blender/blender-3.6.9-linux-x64/blender', 
                        help='Path to the blender executable')
    parser.add_argument('--background', default=False, action='store_true', help='Run blender in background')
    parser.add_argument('--background_mesh_file', type=str, default=None, help='Background mesh file')
    parser.add_argument('--colmap_path', type=str, default=None, help='colmap path')
    parser.add_argument('--camera_fov', type=int, default=90, help='Field of view')
    parser.add_argument('--sampling_rate', type=int, default=50, help='sampling camera poses')
    
    args = parser.parse_args()
    args.colmap_path = '/mnt/hdd/data/Okutama_Action/GS_data/Scenario2/undistorted/sparse/0/'
    args.background_mesh_file = '/mnt/hdd/data/Okutama_Action/GS_data/Scenario2/undistorted/Poisson/mesh_poisson_level10_density9_decim.ply'
    main(args)