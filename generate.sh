MESH_FILE=/mnt/hdd/data/Okutama_Action/GS_data/Scenario2/undistorted/Poisson/mesh_poisson_level10_density9.ply
ACTOR_TEMPLATE_PATH=/mnt/hdd/data/SynBody/SMPL-XL-1000-fbx
ACTOR_INFO_FILE=./data/s2_scenario1.json
OUTPUT_PATH=/mnt/hdd/code/human_data_generation/xrfeitoria/output/waypoint
ACTOR_SCALE=0.125
altitude=2.0
offsetX=0
offsetY=0
SEQ_NAME=altitude${altitude}_offsetX${offsetX}_offsetY${offsetY}_scale${ACTOR_SCALE}

# python data_generation_from_waypoint.py \
#  --actor_template_path $ACTOR_TEMPLATE_PATH --actor_info_file $ACTOR_INFO_FILE \
#  --output_path $OUTPUT_PATH --altitude $altitude --offsetX $offsetX --offsetY $offsetY \
#  --sequence_name $SEQ_NAME --actor_scale_factor $ACTOR_SCALE --background
# # --background_mesh_file $MESH_FILE 

TRAJ_FILE=$OUTPUT_PATH/$SEQ_NAME/camera_trajectory.json
OUTPUT_FOLDER=/mnt/hdd/code/human_data_generation/xrfeitoria/output/waypoint/$SEQ_NAME/background
MODEL_PATH=/mnt/hdd/code/gaussian_splatting/Semantic-GS/gaussian-splatting/output/okutama/Scenario2
ITER=30000

# mkdir -p $OUTPUT_FOLDER
# cd /mnt/hdd/code/gaussian_splatting/Semantic-GS/gaussian-splatting
# python render_with_pose.py --trajectory_file $TRAJ_FILE  --output_folder $OUTPUT_FOLDER \
#  --model_path $MODEL_PATH --iteration $ITER

OUT_FOLDER=/mnt/hdd/code/human_data_generation/xrfeitoria/output/waypoint/$SEQ_NAME
python compose.py --output_path $OUT_FOLDER --motion_blur_degree 4