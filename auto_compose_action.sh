# conda activate yolov8

current_dir=$(pwd)
echo "Current directory: $current_dir"


SPLIT_LIST=(
    # Drone1_Noon_1_2_2 
    # Drone1_Noon_1_2_4 
    # Drone1_Noon_1_2_9 
    Drone2_Noon_2_2_2 
    # Drone2_Noon_2_2_4 
    # Drone2_Noon_2_2_9
)

MOVING_BG_SPLIT_LIST=(
    '-1'
)

# SPLIT_LIST=(
#     # Drone1_Morning_1_1_1 
#     # Drone1_Morning_1_1_4 
#     # Drone1_Morning_1_1_7 
#     # Drone2_Morning_2_1_1
#     # Drone2_Morning_2_1_7
#     # Drone2_Morning_2_1_10
# )

# MOVING_BG_SPLIT_LIST=(
#     # '1 2 3' 
#     # '-1' 
#     # '-1' 
#     # '2 3' # Drone2_Morning_2_1_7
#     # '-1' 
#     # '1' # Drone2_Morning_2_1_10
# )

# how to get index of an element in an array in bash



for i in "${!SPLIT_LIST[@]}"
do 
    split=${SPLIT_LIST[$i]}
    MOVING_BG_SPLIT=${MOVING_BG_SPLIT_LIST[$i]}

    echo "Processing $split"
    echo "Moving BG Split: $MOVING_BG_SPLIT"

    SEQ_NAME='auto_'$split'_alti0'
    OUTPUT_PATH=$current_dir/output/S2_orig_$split/$SEQ_NAME
    # OUTPUT_PATH=$current_dir/output/S3_$split/$SEQ_NAME
    # OUTPUT_PATH=$current_dir/output/S2_vis_$split/$SEQ_NAME
    # OUTPUT_PATH=$current_dir/output/S3_v3_$split/$SEQ_NAME
    
    echo "Composing action for $OUTPUT_PATH"
    python compose_with_shadow_action.py --output_path $OUTPUT_PATH --motion_blur_degree 3 \
        --moving_background --moving_bg_split $MOVING_BG_SPLIT --static_camera
done
