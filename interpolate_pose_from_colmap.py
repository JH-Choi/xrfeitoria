import math
import numpy as np 
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from xrfeitoria.utils.colmap_utils import read_model, convert_to_blender_coord, qvec2rotmat
from xrfeitoria.utils.camera_utils import quaternion_slerp

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


# Load Colmap data
root_path = Path('/mnt/hdd/data/Okutama_Action/Yonghan_data/okutama_n50_Morning')
background_mesh_file = root_path / 'PoissonMeshes_aligned' / 'fused_sor_lod8.ply'
colmap_path = root_path / 'colmap_aligned'
fov = 90

cameras, images, points3D = read_model(str(colmap_path), ext='.bin')
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
sort_image_id = np.argsort(image_names)

# Load Background Current mesh 
xf_runner.utils.import_file(file_path=background_mesh_file)
print('Load background mesh')


steps = 10
ts = np.linspace(0, 1, steps)

poses = []
tot_quats = []
tot_trans = []
# for idx in range(sort_image_id.shape[0] - 1):
for idx in range(3):
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

sequence_name = 'MySequence'
frame_num = num_image
idx = 0 
with xf_runner.Sequence.new(seq_name=sequence_name, seq_length=frame_num, replace=True) as seq:
    for qvec_w2c, tvec_w2c in zip(tot_quats, tot_trans):
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
            fov=fov,
        )

        # use the `camera` in level to render
        seq.use_camera(camera=static_camera)
        idx+=1
    
    pdb.set_trace()
 





