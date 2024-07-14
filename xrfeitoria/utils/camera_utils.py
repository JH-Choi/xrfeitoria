# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Camera transformation helper code.
"""

import math
import numpy as np
from typing import Optional 

# https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/cameras/camera_utils.py

_EPS = np.finfo(float).eps * 4.0


def unit_vector(data, axis: Optional[int] = None) -> np.ndarray:
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    Args:
        axis: the axis along which to normalize into unit vector
        out: where to write out the data to. If None, returns a new np ndarray
    """
    data = np.array(data, dtype=np.float64, copy=True)
    if data.ndim == 1:
        data /= math.sqrt(np.dot(data, data))
        return data
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    return data


def quaternion_slerp(quat0, quat1, fraction: float, spin: int = 0, shortestpath: bool = True) -> np.ndarray:
    """Return spherical linear interpolation between two quaternions.
    Args:
        quat0: first quaternion
        quat1: second quaternion
        fraction: how much to interpolate between quat0 vs quat1 (if 0, closer to quat0; if 1, closer to quat1)
        spin: how much of an additional spin to place on the interpolation
        shortestpath: whether to return the short or long path to rotation
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if q0 is None or q1 is None:
        raise ValueError("Input quaternions invalid.")
    if fraction == 0.0:
        return q0
    if fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0

def get_rotation_matrix(location, actors_center):
    # Calculate direction vector from actors center to location
    direction = np.array(location) - np.array(actors_center)
    direction /= np.linalg.norm(direction)

    # Choose an arbitrary up vector
    up = np.array([0, 0, 1])  # You may need to adjust this based on your coordinate system

    # Calculate right vector
    right = np.cross(direction, up)
    right /= np.linalg.norm(right)

    # Recalculate up vector to ensure orthogonality
    up = np.cross(right, direction)

    # Construct rotation matrix
    rotation_matrix = np.array([right, up, -direction])

    return rotation_matrix

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

# https://github.com/ventusff/neurecon/blob/main/tools/render_view.py
def normalize(vec, axis=-1):
    return vec / (np.linalg.norm(vec, axis=axis, keepdims=True) + 1e-9)

def view_matrix(
    forward: np.ndarray, 
    up: np.ndarray,
    cam_location: np.ndarray):
    rot_z = normalize(forward)
    rot_x = normalize(np.cross(up, rot_z))
    rot_y = normalize(np.cross(rot_z, rot_x))
    mat = np.stack((rot_x, rot_y, rot_z, cam_location), axis=-1)
    hom_vec = np.array([[0., 0., 0., 1.]])
    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])
    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat

def look_at(
    cam_location: np.ndarray, 
    point: np.ndarray, 
    # up=np.array([0., -1., 0.])          # openCV convention
    up=np.array([0., 1., 0.])         # openGL convention
    ):
    # Cam points in positive z direction
    # forward = normalize(point - cam_location)     # openCV convention
    forward = normalize(cam_location - point)   # openGL convention
    return view_matrix(forward, up, cam_location)
