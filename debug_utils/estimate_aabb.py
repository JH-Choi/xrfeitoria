import os
import sys 
import numpy as np 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import trimesh
import xrfeitoria as xf
from xrfeitoria.utils.colmap_utils import read_model, convert_to_blender_coord, qvec2rotmat, read_points3D_binary


colmap_path = '/mnt/hdd/data/Okutama_Action/GS_data/Scenario2/undistorted/sparse/0'

# # use pts3d to estimate aabb
# point_file = os.path.join(colmap_path, 'points3D.bin')
# ptsdata = read_points3D_binary(point_file)
# ptskeys = np.array(sorted(ptsdata.keys()))
# pts3d = np.array([ptsdata[k].xyz for k in ptskeys]) # [M, 3]

# print('pts aabb')
# print(np.min(pts3d, axis=0))
# print(np.max(pts3d, axis=0))

# use mesh to estimate aabb
mesh_path = '/mnt/hdd/data/Okutama_Action/GS_data/Scenario2/undistorted/Poisson/mesh_poisson_level10_density9.ply'
# mesh = trimesh.load(mesh_path)
