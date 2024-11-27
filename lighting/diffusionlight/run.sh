# conda activate diffusionlight
# pip install huggingface_hub==0.24.0

# python inpaint.py --dataset example --output_dir output
# python ball2envmap.py --ball_dir output/square --envmap_dir output/envmap
# python exposure2hdr.py --input_dir output/envmap --output_dir output/hdr


SCENE_NAME=Scenario2
python inpaint.py --dataset example/$SCENE_NAME --output_dir output/$SCENE_NAME
python ball2envmap.py --ball_dir output/$SCENE_NAME/square --envmap_dir output/$SCENE_NAME/envmap
python exposure2hdr.py --input_dir output/$SCENE_NAME/envmap --output_dir output/$SCENE_NAME/hdr
