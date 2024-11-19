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
# Main function (currently from rb_sim_rendering.py)
#########################################################
def run_blender_render(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    blender_cache_dir = config['blender_cache_dir']
 

if __name__ == "__main__":
    parser = ArgumentParserForBlender()
    parser.add_argument('--input_config_path', type=str, default='')
    args = parser.parse_args()
    run_blender_render(args.input_config_path)