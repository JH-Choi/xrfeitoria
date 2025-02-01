# MESH_FILE=/mnt/hdd/data/Okutama_Action/GS_data/Scenario2/undistorted/Poisson/mesh_poisson_level10_density9.ply
MESH_FILE=/mnt/hdd/code/outdoor_relighting/PGSR/output/okutama_r2_wg_mip/Scenario2/mesh_maxdepth10_vox0.01/tsdf_fusion_post_deci.ply
ACTOR_TEMPLATE_PATH=/mnt/hdd/data/SynBody/SMPL-XL-1000-fbx
ACTOR_INFO_FILE=./data/s2_scenario1.json
# OUTPUT_PATH=/mnt/hdd/code/human_data_generation/xrfeitoria/output/waypoint
OUTPUT_PATH=/mnt/hdd/code/human_data_generation/xrfeitoria/output/waypoint_hdr_plane2
ACTOR_SCALE=0.125
altitude=2.0
offsetX=0
offsetY=0
SEQ_NAME=altitude${altitude}_offsetX${offsetX}_offsetY${offsetY}_scale${ACTOR_SCALE}
HDRI_FILE=/mnt/hdd/code/Lighting/DiffusionLight/output/Scenario2/hdr/Drone1_Noon_1_2_2_1090.exr
# HDRI_FILE=./output/dry_orchard_meadow_4k.exr

python data_generation_from_waypoint.py \
 --actor_template_path $ACTOR_TEMPLATE_PATH --actor_info_file $ACTOR_INFO_FILE \
 --output_path $OUTPUT_PATH --altitude $altitude --offsetX $offsetX --offsetY $offsetY \
 --sequence_name $SEQ_NAME --actor_scale_factor $ACTOR_SCALE --hdri_path $HDRI_FILE --use_plane \
 --background_mesh_file $MESH_FILE 
# --background

# TRAJ_FILE=$OUTPUT_PATH/$SEQ_NAME/camera_trajectory.json
# OUTPUT_FOLDER=/mnt/hdd/code/human_data_generation/xrfeitoria/output/waypoint/$SEQ_NAME/background
# MODEL_PATH=/mnt/hdd/code/gaussian_splatting/Semantic-GS/gaussian-splatting/output/okutama/Scenario2
# ITER=30000

# mkdir -p $OUTPUT_FOLDER
# cd /mnt/hdd/code/gaussian_splatting/Semantic-GS/gaussian-splatting
# python render_with_pose.py --trajectory_file $TRAJ_FILE  --output_folder $OUTPUT_FOLDER \
#  --model_path $MODEL_PATH --iteration $ITER

# OUT_FOLDER=/mnt/hdd/code/human_data_generation/xrfeitoria/output/waypoint/$SEQ_NAME
# python compose.py --output_path $OUT_FOLDER --motion_blur_degree 4

# ln -s $OUT_FOLDER /mnt/hdd/data/Okutama_Action/yolov8_Detection/altitude2.0_offsetX0_offsetY0_scale0.125