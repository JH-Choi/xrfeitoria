# conda activate xrfeitoria

# Scenario2 => xcoord [-3, 1] / ycoord [-1, 4]
# Scenario3 => xcoord [-1, 5] / ycoord [-1, 5]

COLMAP_PATH=/mnt/hdd/data/Okutama_Action/GS_data/Scenario2/undistorted/sparse/0
MESH_FILE=/mnt/hdd/code/outdoor_relighting/PGSR/output/okutama_r2_wg_mip/Scenario2/mesh_maxdepth10_vox0.01/tsdf_fusion_post_deci.ply
ACTOR_TEMPLATE_PATH=/mnt/hdd/data/SynBody/SMPL-XL-1000-fbx
OUTPUT_PATH=/mnt/hdd/code/human_data_generation/xrfeitoria/output/auto_hdr_plane2
HDRI_FILE=/mnt/hdd/code/Lighting/DiffusionLight/output/Scenario2_tmp/hdr/Drone1_Noon_1_2_2_1090.exr
MOTION_PATH=/mnt/hdd/data/AMASS/SMPL-X_N
num_actors=8
zcoord=-3
split=Drone1_Noon_1_2_2
SEQ_NAME=tmp


python auto_generation_from_colmap.py \
 --colmap_path $COLMAP_PATH --split $split \
 --sequence_name $SEQ_NAME --hdri_path $HDRI_FILE \
 --split $split \
 --actor_template_path $ACTOR_TEMPLATE_PATH --motion_path $MOTION_PATH --num_actors $num_actors \
 --area_xcoord -3 1 --area_ycoord -1 4 \
 --output_path $OUTPUT_PATH \
 --background_mesh_file $MESH_FILE \
#  --use_plane \
# --background

