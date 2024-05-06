import json 
import pdb

# Load the motion data from the JSON file
with open('okutama_actor.json', 'r') as json_file:
    motion_data = json.load(json_file)

# Access the loaded data
for motion_name, motion_info in motion_data.items():
    print("Motion:", motion_name)
    print("Motion Paths:", motion_info["motion_paths"])
    print("Origins:", motion_info["origins"])
    print("Rotation:", motion_info["rotation"])
    pdb.set_trace()
    