# conda activate pgsr

current_dir=$(pwd)
echo "Current directory: $current_dir"

# # split=Drone1_Noon_1_2_9
# # split=Drone2_Noon_2_2_9
# # split=Drone1_Noon_1_2_4
# split=Drone2_Noon_2_2_4
# # split=Drone2_Noon_2_2_2
# SEQ_NAME='auto_'$split'_alti0'
# # OUTPUT_PATH=$current_dir/output/S2_Drone1_Noon_1_2_2/$SEQ_NAME
# OUTPUT_PATH=$current_dir/output/S2_$split/$SEQ_NAME

# TRAJ_FILE=$OUTPUT_PATH/camera_trajectory.json
# BG_OUTPUT_FOLDER=$OUTPUT_PATH/background

# MOVING_TRAJ_FILE=$OUTPUT_PATH/moving_camera_trajectory.json
# MOVING_BG_OUTPUT_FOLDER=$OUTPUT_PATH/moving_background
# # MODEL_PATH=/mnt/hdd/code/outdoor_relighting/PGSR/output/iccv25/okutama/Scenario2/pgsr_wg_mip_v2_wmask
# MODEL_PATH=/mnt/hdd/code/outdoor_relighting/PGSR/output/iccv25/okutama/Scenario2/pgsr_wg_mip_v2_refinemask

# mkdir -p $BG_OUTPUT_FOLDER
# cd /mnt/hdd/code/outdoor_relighting/PGSR
# CKPT_FILE=$MODEL_PATH/chkpnt30000.pth
# render_type=render

# python render_wg_mip_v2_with_pose.py --trajectory_file $TRAJ_FILE  --output_folder $BG_OUTPUT_FOLDER \
#  --model_path $MODEL_PATH \
#  --render_type $render_type --start_checkpoint $CKPT_FILE  --appearance_enabled \
#  --subfolder_name $split

# python render_wg_mip_v2_with_pose.py --trajectory_file $MOVING_TRAJ_FILE  --output_folder $MOVING_BG_OUTPUT_FOLDER \
#  --model_path $MODEL_PATH \
#  --render_type $render_type --start_checkpoint $CKPT_FILE  --appearance_enabled \
#  --subfolder_name $split

# cd $current_dir
# # python compose.py --output_path $OUTPUT_PATH --motion_blur_degree 3
# python compose_with_shadow.py --output_path $OUTPUT_PATH --motion_blur_degree 3 --moving_background


######################################################################################
######################################################################################

# SPLIT_LIST=(
#     Drone1_Noon_1_2_2 
#     Drone1_Noon_1_2_4 
#     Drone1_Noon_1_2_9 
#     Drone2_Noon_2_2_2 
#     Drone2_Noon_2_2_4 
#     Drone2_Noon_2_2_9
# )

# MOVING_BG_SPLIT_LIST=(
# )

SPLIT_LIST=(
    Drone1_Morning_1_1_1 
    Drone1_Morning_1_1_4 
    Drone1_Morning_1_1_7 
    Drone2_Morning_2_1_1
    Drone2_Morning_2_1_7
    Drone2_Morning_2_1_10
)

MOVING_BG_SPLIT_LIST=(
    '1 3 5 8 9 10 11 15 16' 
    '-1' 
    '-1' 
    '2 3 4 5 6 8 9 11 13 14 15 18 19' 
    '-1' 
    '6 7 8' # Drone2_Morning_2_1_10
)

# how to get index of an element in an array in bash



for i in "${!SPLIT_LIST[@]}"
do 
    split=${SPLIT_LIST[$i]}
    MOVING_BG_SPLIT=${MOVING_BG_SPLIT_LIST[$i]}

    echo "Processing $split"
    echo "Moving BG Split: $MOVING_BG_SPLIT"

    SEQ_NAME='auto_'$split'_alti0'
    # OUTPUT_PATH=$current_dir/output/S2_$split/$SEQ_NAME
    OUTPUT_PATH=$current_dir/output/S3_$split/$SEQ_NAME

    TRAJ_FILE=$OUTPUT_PATH/camera_trajectory.json
    BG_OUTPUT_FOLDER=$OUTPUT_PATH/background

    # MOVING_TRAJ_FILE=$OUTPUT_PATH/moving_camera_trajectory.json
    # MOVING_BG_OUTPUT_FOLDER=$OUTPUT_PATH/moving_background
    
    ###  Define Model Path
    # MODEL_PATH=/mnt/hdd/code/outdoor_relighting/PGSR/output/iccv25/okutama/Scenario2/pgsr_wg_mip_v2_wmask
    # MODEL_PATH=/mnt/hdd/code/outdoor_relighting/PGSR/output/iccv25/okutama/Scenario2/pgsr_wg_mip_v2_refinemask
    MODEL_PATH=/mnt/hdd/code/outdoor_relighting/PGSR/output/iccv25/okutama/Scenario3/pgsr_wg_mip_v2_wmask
    # MODEL_PATH=/mnt/hdd/code/outdoor_relighting/PGSR/output/iccv25/okutama/Scenario3/pgsr_wg_mip_v2_refinemask

    ### Rendering
    mkdir -p $BG_OUTPUT_FOLDER
    cd /mnt/hdd/code/outdoor_relighting/PGSR
    CKPT_FILE=$MODEL_PATH/chkpnt30000.pth
    render_type=render

    # python render_wg_mip_v2_with_pose.py --trajectory_file $TRAJ_FILE  --output_folder $BG_OUTPUT_FOLDER \
    # --model_path $MODEL_PATH \
    # --render_type $render_type --start_checkpoint $CKPT_FILE  --appearance_enabled \
    # --subfolder_name $split

    # # Find the trajectory file
    # MOVING_TRAJ_FILES=$(find "$OUTPUT_PATH" -type f -name "moving_camera_*_trajectory.json")

    # for MOVING_TRAJ_FILE in $MOVING_TRAJ_FILES; do
    #     BASE_NAME=$(basename "$MOVING_TRAJ_FILE" .json)
    #     BASE_NAME=${BASE_NAME//_trajectory/}
    #     MOVING_BG_OUTPUT_FOLDER=$OUTPUT_PATH/background_$BASE_NAME

    #     mkdir -p $MOVING_BG_OUTPUT_FOLDER
    #     echo "Processing $MOVING_TRAJ_FILE..."

    #     python render_wg_mip_v2_with_pose.py --trajectory_file $MOVING_TRAJ_FILE  --output_folder $MOVING_BG_OUTPUT_FOLDER \
    #     --model_path $MODEL_PATH \
    #     --render_type $render_type --start_checkpoint $CKPT_FILE  --appearance_enabled \
    #     --subfolder_name $split
    # done

    ### Compose Data
    cd $current_dir
    # # python compose.py --output_path $OUTPUT_PATH --motion_blur_degree 3
    python compose_with_shadow.py --output_path $OUTPUT_PATH --motion_blur_degree 3 \
        --moving_background --moving_bg_split $MOVING_BG_SPLIT
done
