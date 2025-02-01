import bpy
from xrfeitoria.rpc import remote_blender
from xrfeitoria.utils.anim import load_amass_motion

@remote_blender()
def apply_scale(actor_name: str, scale_factor: float):

    obj = bpy.data.objects.get(actor_name)
    # bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    # bpy.data.objects[actor_name].select_set(True)
    bpy.ops.object.transform_apply(scale=True)
    obj.scale.x *= scale_factor
    obj.scale.y *= scale_factor
    obj.scale.z *= scale_factor


@remote_blender()
def generate_plane(actor_name: str, size):
    plane = bpy.data.objects.get(actor_name)
    plane.select_set(True)
    plane.scale = (size, size, 1)
    plane.is_shadow_catcher = True


def read_motion(motion_path, fps=None, start_frame=None, end_frame=None):
    # fps:  convert the motion from 120fps (amass) to 30fps
    #  cut the motion to 10 frames, for demonstration purpose
    motion = load_amass_motion(motion_path)  # modify this to motion file in absolute path
    if fps is not None:
        motion.convert_fps(fps)  # convert the motion from 120fps (amass) to 30fps
    motion.cut_motion(start_frame=start_frame, end_frame=end_frame)
    motion_data = motion.get_motion_data()
    n_frames = motion.n_frames
    return motion_data, n_frames

