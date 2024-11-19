import pickle
import numpy as np
import os
import sys
import bpy
import math
import shutil
import json
import time
from mathutils import Vector, Matrix
import argparse
import glob
import colorsys
import bmesh
from mathutils.bvhtree import BVHTree


"""
Blender python script for rendering all visual effects.
"""


context = bpy.context
scene = context.scene
render = scene.render


#########################################################
# Argument parser for blender: https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
#########################################################
class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


#########################################################
# Blender scene setup
#########################################################
def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def setup_blender_env(img_width, img_height):

    reset_scene()

    # Set render engine and parameters
    render.engine = 'CYCLES'
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = img_width
    render.resolution_y = img_height
    render.resolution_percentage = 100

    scene.cycles.device = "GPU"
    scene.cycles.preview_samples = 64
    scene.cycles.samples = 64  # 32 for testing, 256 or higher 512 for final
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.cycles.film_exposure = 2.0

    # Set the device_type (from Zhihao's code, not sure why specify this)
    preferences = context.preferences
    preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA" # or "OPENCL"

    # get_devices() to let Blender detects GPU device
    preferences.addons["cycles"].preferences.get_devices()
    print(preferences.addons["cycles"].preferences.compute_device_type)
    for d in preferences.addons["cycles"].preferences.devices:
        d["use"] = 1 # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

#########################################################
# Blender camera setup
#########################################################
def create_camera_list(c2w, K):
    """
    Create a list of camera parameters

    Args:
        c2w: (N, 4, 4) camera to world transform
        K: (3, 3) or (N, 3, 3) camera intrinsic matrix
    """
    cam_list = []
    for i in range(len(c2w)):
        pose = c2w[i].reshape(-1, 4)
        if len(K.shape) == 3:
            cam_list.append({'c2w': pose, 'K': K[i]})
        else:
            cam_list.append({'c2w': pose, 'K': K})
    return cam_list


def setup_camera():
    # Find a camera in the scene
    cam = None
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            cam = obj
            print("found camera")
            break
    # If no camera is found, create a new one
    if cam is None:
        bpy.ops.object.camera_add()
        cam = bpy.context.object
    # Set the camera as the active camera for the scene
    bpy.context.scene.camera = cam
    return cam


class Camera():
    def __init__(self, im_height, im_width, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.w = im_width
        self.h = im_height
        self.camera = setup_camera()
        
    def set_camera(self, K, c2w):
        self.K = K       # (3, 3)
        self.c2w = c2w   # (3 or 4, 4), camera to world transform
        # original camera model: x: right, y: down, z: forward (OpenCV, COLMAP format)
        # Blender camera model:  x: right, y: up  , z: backward (OpenGL, NeRF format)
        
        self.camera.data.type = 'PERSP'
        self.camera.data.lens_unit = 'FOV'
        f = K[0, 0]
        rad = 2 * np.arctan(self.w/(2 * f))
        self.camera.data.angle = rad
        self.camera.data.sensor_fit = 'HORIZONTAL'  # 'HORIZONTAL' keeps horizontal right (more recommended)

        # f = K[1, 1]
        # rad = 2 * np.arctan(self.h/(2 * f))
        # self.camera.data.angle = rad
        # self.camera.data.sensor_fit = 'VERTICAL'  # 'VERTICAL' keeps vertical right
        
        self.pose = self.transform_pose(c2w)
        self.camera.matrix_world = Matrix(self.pose)
        
    def transform_pose(self, pose):
        '''
        Transform camera-to-world matrix
        Input:  (3 or 4, 4) x: right, y: down, z: forward
        Output: (4, 4)      x: right, y: up  , z: backward
        '''
        pose_bl = np.zeros((4, 4))
        pose_bl[3, 3] = 1
        # camera position remain the same
        pose_bl[:3, 3] = pose[:3, 3] 
        
        R_c2w = pose[:3, :3]
        transform = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ]) 
        R_c2w_bl = R_c2w @ transform
        pose_bl[:3, :3] = R_c2w_bl
        
        return pose_bl

    def initialize_depth_extractor(self):
        bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
        bpy.context.view_layer.cycles.use_denoising = True
        bpy.context.view_layer.cycles.denoising_store_passes = True
        bpy.context.scene.use_nodes = True

        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links

        render_layers = nodes['Render Layers']
        depth_file_output = nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.name = 'File Output Depth'
        depth_file_output.format.file_format = 'OPEN_EXR'
        links.new(render_layers.outputs[2], depth_file_output.inputs[0])

    def render_single_timestep_rgb_and_depth(self, cam_info, FRAME_INDEX, dir_name_rgb='rgb', dir_name_depth='depth'):

        dir_path_rgb = os.path.join(self.out_dir, dir_name_rgb)
        dir_path_depth = os.path.join(self.out_dir, dir_name_depth)
        os.makedirs(dir_path_rgb, exist_ok=True)
        os.makedirs(dir_path_depth, exist_ok=True)

        self.set_camera(cam_info['K'], cam_info['c2w'])

        # Set paths for both RGB and depth outputs
        depth_output_path = os.path.join(dir_path_depth, '{:0>3d}'.format(FRAME_INDEX))
        rgb_output_path = os.path.join(dir_path_rgb, '{:0>3d}.png'.format(FRAME_INDEX))

        # Assuming your Blender setup has nodes named accordingly
        bpy.context.scene.render.filepath = rgb_output_path
        bpy.data.scenes["Scene"].node_tree.nodes["File Output Depth"].base_path = depth_output_path

        bpy.ops.render.render(use_viewport=True, write_still=True)



#########################################################
# Blender lighting setup
#########################################################
def add_env_lighting(env_map_path: str, strength: float = 1.0):
    """
    Add environment lighting to the scene with controllable strength.

    Args:
        env_map_path (str): Path to the environment map.
        strength (float): Strength of the environment map.
    """
    # Ensure that we are using nodes for the world's material
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()

    # Create an environment texture node and load the image
    env = nodes.new('ShaderNodeTexEnvironment')
    env.image = bpy.data.images.load(env_map_path)

    # Create a Background node and set its strength
    background = nodes.new('ShaderNodeBackground')
    background.inputs['Strength'].default_value = strength

    # Create an Output node
    out = nodes.new('ShaderNodeOutputWorld')

    # Link nodes together
    links = world.node_tree.links
    links.new(env.outputs['Color'], background.inputs['Color'])
    links.new(background.outputs['Background'], out.inputs['Surface'])


def add_sun_lighting(strength: float = 1.0, direction=(0, 0, 1)):
    """
    Add a sun light to the scene with controllable strength and direction.

    Args:
        strength (float): Strength of the sun light.
        direction (tuple): Direction of the sun light.
    """
    sun_name = 'Sun'
    sun = bpy.data.objects.get(sun_name)
    if sun is None:
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 0))
        sun = bpy.context.object
        sun.name = sun_name

    direction = Vector(direction)
    direction.normalize()
    rotation = direction.to_track_quat('Z', 'Y').to_euler()
    sun.rotation_euler = rotation
    sun.data.energy = strength


#########################################################
# Object manipulation
#########################################################

def load_object(object_path: str) -> bpy.types.Object:
    """Loads an object asset into the scene."""
    # import the object
    if object_path.endswith(".glb") or object_path.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path, axis_forward='Y', axis_up='Z')
    elif object_path.endswith(".ply"):
        # bpy.ops.import_mesh.ply(filepath=object_path)                             # only used for snap blender
        bpy.ops.wm.ply_import(filepath=object_path, forward_axis='Y', up_axis='Z')  # used for blender 4.0 & snap blender
    elif object_path.endswith(".obj"):
        # bpy.ops.import_scene.obj(filepath=object_path, use_split_objects=False, forward_axis='Y', up_axis='Z')  # only used for snap blender
        bpy.ops.wm.obj_import(filepath=object_path, use_split_objects=False, forward_axis='Y', up_axis='Z')       # used for blender 4.0 & snap blender
    # ##### This part is used for ChatSim assets #####
    # elif object_path.endswith(".blend"):
    #     blend_path = object_path
    #     new_obj_name = 'chatsim_' + blend_path.split('/')[-1].split('.')[0]
    #     model_obj_name = 'Car'                                                  # general names used for all assets in ChatSim
    #     with bpy.data.libraries.load(blend_path) as (data_from, data_to):
    #         data_to.objects = data_from.objects
    #     for obj in data_to.objects:                                             # actually part that import the object
    #         if obj.name == model_obj_name:
    #             bpy.context.collection.objects.link(obj)
    #     if model_obj_name in bpy.data.objects:                                  # rename the object to avoid conflict
    #         imported_object = bpy.data.objects[model_obj_name]
    #         imported_object.name = new_obj_name
    #         print(f"rename {model_obj_name} to {new_obj_name}")
    #     for slot in imported_object.material_slots:                             # rename the material to avoid conflict
    #         material = slot.material
    #         if material:
    #             material.name = new_obj_name + "_" + material.name
    #     return imported_object
    else:
        raise ValueError(f"Unsupported file type: {object_path}")
    new_obj = bpy.context.object
    return new_obj


#########################################################
# Shadow catcher setup
#########################################################
def add_meshes_shadow_catcher(mesh_path=None, is_uv_mesh=False):
    """
    Add entire scene meshes as shadow catcher to the scene

    Args:
        mesh_path: path to the mesh file
        is_uv_mesh: whether the mesh is a UV textured mesh
    """
    # add meshes extracted from NeRF/3DGS as shadow catcher
    if mesh_path is None or not os.path.exists(mesh_path):
        AssertionError('meshes file does not exist')
    mesh = load_object(mesh_path)
    # mesh.is_shadow_catcher = True   # set True for transparent shadow catcher
    mesh.visible_diffuse = False      # prevent meshes light up the scene

    if not is_uv_mesh:
        mesh.visible_glossy = False   # prevent white material from reflecting light
        white_mat = create_white_material()
        if mesh.data.materials:
            mesh.data.materials[0] = white_mat
        else:
            mesh.data.materials.append(white_mat)

    bpy.ops.object.select_all(action="DESELECT")
    return mesh


def add_planar_shadow_catcher(size=10):
    """
    Add a large planar surface as shadow catcher to the scene

    Args:
        size: size of the planar surface
    """
    bpy.ops.mesh.primitive_plane_add(size=1)
    mesh = bpy.context.object

    mesh.scale = (size, size, 1)
    mesh.name =  "floor_plane"

    mesh.visible_glossy = False   # prevent white material from reflecting light
    white_mat = create_white_material()
    if mesh.data.materials:
        mesh.data.materials[0] = white_mat
    else:
        mesh.data.materials.append(white_mat)

    bpy.ops.object.select_all(action="DESELECT")
    return mesh




#########################################################
# Materials
#########################################################
def create_white_material():
    mat = bpy.data.materials.new(name="WhiteMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
    bsdf.inputs["Metallic"].default_value = 0.0
    # bsdf.inputs["Specular"].default_value = 0.0  # issue: https://github.com/ross-g/io_pdx_mesh/issues/86
    bsdf.inputs[7].default_value = 0.0   # Specular
    bsdf.inputs["Roughness"].default_value = 1.0
    return mat


#########################################################
# Global variables
#########################################################
all_object_info = {}           # object_id -> object_info from insert_object_info
all_object_dict = {}           # object_id -> blender object
object_list = []               # list of foreground objects (ex: blender assets or 3dgs objects with modified materials)
human_list = []          # list of 3dgs objects (ex: 3dgs objects with original materials)
object_3dgs_list = []          # list of 3dgs objects (ex: 3dgs objects with original materials)
object_3dgs_scale_dict = {}    # scale of 3dgs objects (keep the scale here since apply_transform will change the scale to 1.0 for the sake of rigid body simulation)
smoke_domain_dict = {}         # object_id -> smoke domain
fire_object_id_list = []       # list of object_id that has fire
smoke_object_id_list = []      # list of object_id that has smoke
fracture_object_list = []      # list of fracturable objects (keep track if collision happens)
all_events_dict = {}           # object_id -> list of events
debris_object_list = []        # store the debris generated from fracture

smoke_proxy_obj_dict = {}      # object_id -> smoke domain for proxy object
all_3dgs_object_names = []     # list of names of all 3dgs objects (used for custom post-filtering after creating fractures)

fluid_domain_dict = {}             # object_id -> fluid domain
melting_related_object_dict = {}   # object_id -> fluid related object
melting_object_list = []           # list of object_id that has melting effect
fluid_toggle_on_list = []             # list of object_id that has fluid simulation toggle on

# COLLISION_MARGIN = 0.001
# DOMAIN_HEIGHT = 8.0
# CACHE_DIR = None 
# # CACHE_DIR = '/tmp/smoke_cache'  # default cache directory for smoke simulation


#########################################################
# Main function (currently from rb_sim_rendering.py)
#########################################################
def run_blender_render(config_path):
    print(config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
    blender_cache_dir = config['blender_cache_dir']
    h, w = config['im_height'], config['im_width']
    # K = np.array(config['K'])
    # c2w = np.array(config['c2w'])

    if "output_dir_name" in config:
        output_dir = os.path.join(blender_cache_dir, config["output_dir_name"])
    else:
        output_dir = os.path.join(blender_cache_dir, 'blend_results')
    os.makedirs(output_dir, exist_ok=True)


    # scene_mesh_path = config['scene_mesh_path']
    # is_uv_mesh = config['is_uv_mesh']

    # # anti-aliasing rendering
    # upscale = 2.0
    # w = int(w * upscale)
    # h = int(h * upscale)
    # if len(K.shape) == 2:
    #     K[0, 0] *= upscale
    #     K[1, 1] *= upscale
    #     K[0, 2] *= upscale
    #     K[1, 2] *= upscale
    # else:
    #     for i in range(len(K)):
    #         K[i][0, 0] *= upscale
    #         K[i][1, 1] *= upscale
    #         K[i][0, 2] *= upscale
    #         K[i][1, 2] *= upscale

    setup_blender_env(w, h)

    # scene_mesh = add_meshes_shadow_catcher(scene_mesh_path, is_uv_mesh)

    # Add a large plane for waymo scene (if required)
    # planar_mesh = None
    planar_mesh = add_planar_shadow_catcher(size=50)


    # Add environment map
    # add_env_lighting(global_env_map_path, strength=0.6)     # outdoor scene

    # print("=====> Add sun lighting for waymo scene")
    # sun_dir = np.array(config['sun_dir'])
    # add_sun_lighting(1.0, sun_dir)


    cam = Camera(h, w, output_dir)
    # cam_list = create_camera_list(c2w, K)

    scene.frame_start = 1
    # if config["render_type"] == 'SINGLE_VIEW':
    #     scene.frame_end = config['num_frames']
    # else:
    #     scene.frame_end = len(c2w)
    # print("frame start: {}, frame end: {}".format(scene.frame_start, scene.frame_end))

    # for obj_info in insert_object_info:
    #     all_object_info[obj_info['object_id']] = obj_info

    # # setup the events
    # if 'events' in config:
    #     for event in config['events']:
    #         event_list = event_parser(event)
    #         for obj_id, action, frame_num in event_list:
    #             if obj_id not in all_events_dict:
    #                 all_events_dict[obj_id] = []
    #             all_events_dict[obj_id].append((obj_id, action, frame_num))


    bpy.context.view_layer.update()     # Update the scene

    # cam.initialize_depth_extractor()  # initialize once

    # for FRAME_INDEX in range(scene.frame_start, scene.frame_end + 1):
    for FRAME_INDEX in range(0, 1):
        print("Frame index: ", FRAME_INDEX)
        print("===== All object names: ", [obj.name for obj in all_object_dict.values()])
        print("===== Object names: ", [obj.name for obj in object_list])
        # print("===== 3DGS object names: ", [obj.name for obj in object_3dgs_list])
        # print("===== Fracture objects: ", [obj.name for obj in fracture_object_list])
        # print("===== Debris objects: ", [obj.name for obj in debris_object_list])

        bpy.context.view_layer.update()     # Ensure the scene is fully updated

        # Step 1: render only inserted objects



if __name__ == "__main__":
    parser = ArgumentParserForBlender()
    parser.add_argument('--input_config_path', type=str, default='')
    args = parser.parse_args()
    run_blender_render(args.input_config_path)