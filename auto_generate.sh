# conda activate xrfeitoria

# Scenario2 => xcoord [-3, 1] / ycoord [-1, 4]
# Scenario3 => xcoord [-1, 5] / ycoord [-1, 5]

current_dir=$(pwd)
echo "Current directory: $current_dir"


PID=$(ps -a | grep blender | awk '{print $1}')

if [ -n "$PID" ]; then
    echo "Killing Blender process with PID: $PID"
    kill "$PID"
else
    echo "No Blender process found."
fi

# Drone1_Noon_1_2_2 : xcoord [-2, 1] / ycoord [-1, 4]
# Drone2_Noon_2_2_2 : xcoord [-0.5, 0.5] / ycoord [-1, 5]
# Drone1_Noon_1_2_4 : xcoord [-2, 0] / ycoord [-1, 4]
# Drone2_Noon_2_2_4 : xcoord [-5, 1] / ycoord [0, 0.5]
# Drone1_Noon_1_2_9 : xcoord [-0.5, -4.5] / ycoord [5, 10]
# Drone2_Noon_2_2_9 : xcoord [-0.5, -4.5] / ycoord [5, 10]

# Drone1_Morning_1_1_1 : xcoord [-1, 10] / ycoord [0, 10]
# Drone2_Morning_2_1_1 : xcoord [-1, 5] / ycoord [-0.1, 2]
# Drone1_Morning_1_1_4 : xcoord [-1, 0.5] / ycoord [3, 10]
# Drone1_Morning_1_1_7 : xcoord [-1.5, 7] / ycoord [0, 10]
# Drone2_Morning_2_1_7 : xcoord [-1, 10] / ycoord [0, 10]
# Drone2_Morning_2_1_10 : xcoord [-1, 10] / ycoord [0, 10]


# split=Drone1_Noon_1_2_2
# xcoord='-2 1'
# ycoord='-1 4'

split=Drone2_Noon_2_2_2
xcoord='-0.5 0.5'
ycoord='-1 5'


# split=Drone1_Morning_1_1_1
# xcoord='-1 10'
# ycoord='0 10'

# split=Drone1_Morning_1_1_4
# xcoord='-1 0.5'
# ycoord='3 10'

# split=Drone1_Morning_1_1_7
# xcoord='-1.5 7'
# ycoord='0 10'

# split=Drone2_Morning_2_1_1
# xcoord='-1 5'
# ycoord='-1 5'

# split=Drone2_Morning_2_1_7
# xcoord='-1 10'
# ycoord='0 10'

# split=Drone2_Morning_2_1_10
# xcoord='-1 10'
# ycoord='0 10'



ENGINE_EXE_PATH=./dataset/blender-3.6.9-linux-x64/blender
COLMAP_PATH=./dataset/Okutama_Action/GS_data/Scenario2/undistorted/sparse/0
MESH_FILE=./dataset/Okutama_Action/GS_data/Scenario2/undistorted/mesh_maxdepth10_vox0.01/tsdf_fusion_post_deci.ply
# COLMAP_PATH=./dataset/Okutama_Action/GS_data/Scenario3/undistorted/sparse/0
# MESH_FILE=./dataset/Okutama_Action/GS_data/Scenario3/undistorted/mesh_maxdepth10_vox0.01/tsdf_fusion_post_deci.ply
ACTOR_TEMPLATE_PATH=./dataset/SynBody/SMPL-XL-1000-fbx
MOTION_PATH=./dataset/AMASS/SMPL-X_N
num_actors=10 # 8 / 10 (0.3, 0.4, 0.1, 0.1, 0.1) / 20 (0.4, 0.4, 0.05, 0.05, 0.1)
zcoord=-3 # -3
scale_factor=0.135 # Scenairo2 : 0.135 / Scenario3 : 0.19 
# scale_factor=0.19 # Scenairo2 : 0.135 / Scenario3 : 0.19 
HDRI_FILE=/mnt/hdd/code/Lighting/DiffusionLight/output/Scenario2_Drone1_Noon_1_2_2/hdr/Drone1_Noon_1_2_2_1090.exr
# HDRI_FILE=/mnt/hdd/code/Lighting/DiffusionLight/output/Scenario3_Drone1_Morning_1_1_10/hdr/Drone1_Morning_1_1_10_248.exr
# split=Drone1_Noon_1_2_4
# split=Drone2_Noon_2_2_4
# split=Drone2_Noon_2_1_10
altitude=0 # 0 
SEQ_NAME='auto_'$split'_alti'$altitude
# OUTPUT_PATH=$current_dir/output/S2_vis_$split
OUTPUT_PATH=$current_dir/output/S2_orig_$split
# OUTPUT_PATH=$current_dir/output/S4_v3_$split
# OUTPUT_PATH=$current_dir/output/S3_$split

python auto_generation_from_colmap.py \
 --engine_exec_path $ENGINE_EXE_PATH \
 --colmap_path $COLMAP_PATH --split $split \
 --sequence_name $SEQ_NAME \
 --split $split \
 --actor_scale_factor $scale_factor \
 --actor_template_path $ACTOR_TEMPLATE_PATH --motion_path $MOTION_PATH --num_actors $num_actors \
 --area_xcoord $xcoord --area_ycoord $ycoord \
 --actor_parition_ratio 0.3 0.4 0.1 0.1 0.1 \
 --output_path $OUTPUT_PATH \
 --interpolate_rate 1 \
 --resolution 1319 725 \
 --altitude $altitude --hdri_path $HDRI_FILE --moving_camera --zcoord $zcoord \
 --background --use_plane --render_samples 32
#  --background_mesh_file $MESH_FILE \
#  --use_plane --altitude $altitude --hdri_path $HDRI_FILE --moving_camera 
# --background
#  --sequence_name $SEQ_NAME --hdri_path $HDRI_FILE \
#  --background_mesh_file $MESH_FILE \
# --use_plane \
