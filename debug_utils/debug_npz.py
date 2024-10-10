import os
import numpy as np 
import pdb

root_folder = '/mnt/hdd/code/human_data_generation/xrfeitoria/output/blender_waypoint/altitude1.0_offsetX5_offsetY3/vertices/'
file = os.path.join(root_folder, 'People.001.npz')
mesh = np.load(file, allow_pickle=True)
# mesh['verts'] # timeframe, 10511, 3
# mesh['faces'] # timeframe, 20908, 3
