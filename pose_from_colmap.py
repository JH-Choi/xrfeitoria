import math
import json
import xrfeitoria as xf
from pathlib import Path
from xrfeitoria.data_structure.models import RenderPass
from xrfeitoria.data_structure.models import SequenceTransformKey as SeqTransKey
from argparse import ArgumentParser

import re
import numpy as np 
from scipy.spatial.transform import Rotation as R
from xrfeitoria.utils.colmap_utils import read_model, convert_to_blender_coord, qvec2rotmat
from xrfeitoria.utils.motion_utils import apply_scale
import pdb

##### Okutama Action
def extract_sort_key(filename):
    """Extracts numerical components from a filename and returns them as a tuple of integers."""
    parts = re.findall(r'\d+', filename) 
    return tuple(map(int, parts))  # Convert to tuple for proper sorting

##### Archangel
def extract_numbers(filename):
    numbers = list(map(int, re.findall(r'\d+', filename)))  # Extract all numbers as integers
    return numbers  # Sorting will be based on this sequence of numbers


def main(args): 
    exec_path_stem = Path(args.engine_exec_path).stem.lower()
    if 'blender' in exec_path_stem:
        # Open Blender
        render_engine = 'blender'
        xf_runner = xf.init_blender(exec_path=args.engine_exec_path, 
                                    background=False, 
                                    new_process=True)

    # Load Colmap data
    colmap_path = Path(args.colmap_path)

    cameras, images, points3D = read_model(str(colmap_path), ext='.bin')

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
    colmap_data['points3D'] = points3D


    # Load colmap data
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
    # sort_image_id = np.argsort(image_names)

    sorted_indices_and_names = sorted(enumerate(image_names), key=lambda x: extract_sort_key(x[1]))
    sort_image_id = [idx for idx, _ in sorted_indices_and_names]
    # sorted_filenames = [name for _, name in sorted_indices_and_names]
    
    # Check the minimum / maximum of camera locations for each split
    camera_minmax_dict = {}
    for idx, i_id in enumerate(sort_image_id):
        img_name = image_names[i_id]
        scene_name = '_'.join(img_name.split('_')[:-1])
        if scene_name not in camera_minmax_dict:
            camera_minmax_dict[scene_name] = {}
            camera_minmax_dict[scene_name]['min'] = [np.inf, np.inf, np.inf]
            camera_minmax_dict[scene_name]['max'] = [-np.inf, -np.inf, -np.inf]
        tvec_w2c = image_translation[i_id]
        qvec_w2c = image_quaternion[i_id]
        tvec, qvec = convert_to_blender_coord(tvec_w2c, qvec_w2c)
        location = np.array([tvec[0], tvec[1], tvec[2]])

        min_coords = np.minimum(np.array(camera_minmax_dict[scene_name]['min']), location)
        max_coords = np.maximum(np.array(camera_minmax_dict[scene_name]['max']), location)
        camera_minmax_dict[scene_name]['min'] = list(min_coords)
        camera_minmax_dict[scene_name]['max'] = list(max_coords)

    # with open("Scenario3_camera_minmax.json", "w") as json_file:
    #     json.dump(camera_minmax_dict, json_file, indent=4)  # `indent=4` makes it readable

    print(camera_minmax_dict)
    # Load Background Current mesh 
    if args.background_mesh_file is not None:
        xf_runner.utils.import_file(file_path=args.background_mesh_file)
        print('Load background mesh')

    pose_blender_dict = {}  
    sequence_name = 'MySequence'
    frame_num = num_image
    with xf_runner.Sequence.new(seq_name=sequence_name, seq_length=frame_num, replace=True) as seq:
        for idx, i_id in enumerate(sort_image_id):
            if idx % args.sampling_rate != 0:
                continue
            tvec_w2c = image_translation[i_id]
            qvec_w2c = image_quaternion[i_id]
            tvec, qvec = convert_to_blender_coord(tvec_w2c, qvec_w2c)

            location = (tvec[0], tvec[1], tvec[2])
            # qvec in the order [w,x,y,z]
            qvec = np.array([qvec[1], qvec[2], qvec[3], qvec[0]])
            Rot = R.from_quat(qvec)
            rotation = Rot.as_euler('xyz', degrees=True)
            rotation = tuple(r for r in rotation)

            # Add a static camera
            static_camera = seq.spawn_camera(
                camera_name=f'static_camera_{idx}',
                location=location,
                rotation=rotation,
                fov=args.fov,
            )

            apply_scale(f'static_camera_{idx}', scale_factor=args.camera_scale_factor) 
            
            # Write a pose to the dictionary
            pose_blender_dict[f'static_camera_{idx}'] = {}
            pose_blender_dict[f'static_camera_{idx}']['R'] = rotation
            pose_blender_dict[f'static_camera_{idx}']['T'] = location

            # use the `camera` in level to render
            seq.use_camera(camera=static_camera)
        pdb.set_trace()

    cam_file_path = f'{args.split}_pose_from_colmap.json'
    with open(cam_file_path, "w") as outfile:
        json.dump(pose_blender_dict, outfile, indent=True)
    print(f'Results saved to "{cam_file_path}". # of frames: {len(pose_blender_dict["R"])}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--engine_exec_path', type=str, 
                        default='/mnt/hdd/code/blender/blender-3.6.9-linux-x64/blender', 
                        help='Path to the blender executable')
    parser.add_argument('--background', default=False, action='store_true', help='Run blender in background')
    parser.add_argument('--background_mesh_file', type=str, default=None, help='Background mesh file')
    parser.add_argument('--colmap_path', type=str, default=None, help='colmap path')
    parser.add_argument('--fov', type=int, default=90, help='Field of view')
    parser.add_argument('--sampling_rate', type=int, default=5, help='sampling camera poses')
    parser.add_argument('--split', type=str, default=None, help='camera split')
    parser.add_argument('--camera_scale_factor', type=float, default=0.3, help='camera visualization size')
    
    args = parser.parse_args()
    # args.colmap_path = '/mnt/hdd/data/Okutama_Action/GS_data/Scenario2/undistorted/sparse/0/'
    # args.background_mesh_file = '/mnt/hdd/code/outdoor_relighting/PGSR/output/okutama_r2_wg_mip/Scenario2/mesh_maxdepth10_vox0.01/tsdf_fusion_post_deci.ply'
    args.colmap_path = '/mnt/hdd/data/Okutama_Action/GS_data/Scenario3/undistorted/sparse/0/'
    args.background_mesh_file = '/mnt/hdd/code/outdoor_relighting/PGSR/output/okutama_r2_wg_mip/Scenario3/mesh_maxdepth10_vox0.01/tsdf_fusion_post_deci.ply'
    # args.colmap_path = '/mnt/hdd/data/Archangel/Scenario1/undistorted/sparse/0'
    # args.colmap_path = '/mnt/hdd/data/Archangel/Scenario1/sparse_aligned/0'
    # args.background_mesh_file = '/mnt/hdd/data/Okutama_Action/GS_data/Scenario2/undistorted/Poisson/mesh_poisson_level10_density9_decim.ply'
    # args.split = 'Drone2_Noon_2_2_2'
    # args.split = 'Drone1_Noon_1_2_4'
    # args.split = 'Drone1_Morning_1_1_1'
    # args.split = 'Drone1_Noon_1_2_9'
    args.split = 'Drone2_Morning_2_1_10'
    # args.split = 'Drone1_Morning_1_1_7'
    main(args)