""" 
python -m tests.unreal.init_test
"""

import numpy as np
from loguru import logger

from ..utils import __timer__, _init_unreal, set_logger


def camera_test(debug: bool = False, background: bool = False):
    set_logger(debug=debug)
    with _init_unreal(background=background) as xf_runner:
        with __timer__("spawn camera"):
            camera = xf_runner.Camera.spawn_camera(
                name="new cam",
                location=(500, 300, 200),
                rotation=(30, 0, 0),
                fov=90.0,
            )
            assert np.allclose(
                camera.location, (500, 300, 200)
            ), f"location not match, camera.location={camera.location}"
            assert np.allclose(camera.rotation, (30, 0, 0)), f"rotation not match, camera.rotation={camera.rotation}"
            assert np.allclose(camera.fov, 90.0), f"fov not match, camera.fov={camera.fov}"

        with __timer__("set location"):
            camera.location = (100, 200, 300)
            assert np.allclose(
                camera.location, (100, 200, 300)
            ), f"location not match, camera.location={camera.location}"

        with __timer__("set rotation"):
            camera.rotation = (20, 30, 50)
            assert np.allclose(camera.rotation, (20, 30, 50)), f"rotation not match, camera.rotation={camera.rotation}"

        with __timer__("set fov"):
            camera.fov = 60.0
            assert np.allclose(camera.fov, 60.0), f"fov not match, camera.fov={camera.fov}"

        # with __timer__("delete camera"):
        #     camera.delete()

    logger.info("🎉 camera tests passed!")


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--debug", action="store_true")
    args.add_argument("--background", "-b", action="store_true")
    args = args.parse_args()

    camera_test(debug=args.debug, background=args.background)
