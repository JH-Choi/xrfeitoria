```
python convert_mesh_to_origin.py
python data_generation.py 
```

### Asset of motions 
https://github.com/openxrlab/xrfeitoria/issues/11


### How to check global coordinate of mesh vertex in blender 
Edit Mode, Click vertex, and press N button


### Code explanation
data_generation.py 

### Debug Camera Trajectory using Visualization 
* pose_from_colmap.py : visualize COLMAP camera pose
* interpolate_pose_from_colmap.py : interpolate camera pose from colmap
* waypoint_trajectory_generation.py : interpolate camera pose from waypoints 

### Generate Synthetic Data
* data_generation_from_colmap.py : genearate synthetic data using COLMAP poses
* data_generation_from_waypoint.py : genearate synthetic data using camera waypoints 
* data_generation_from_trajectory.py : (clean the code => data_generation_from_waypoint.py) genearate synthetic data from camera trajectory  


### Generate Synthetic Data v2 
Randomly distribute human and its motion 
* auto_generation_from_colmap.py : genearate synthetic data using COLMAP poses

### Scripts for generating synthetic data
* auto_generate.sh: Generate Data
* auto_compose.sh: Scene Composition 
* auto_combine.sh: Combine all Data