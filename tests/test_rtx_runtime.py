# SPDX-License-Identifier: Apache-2.0
"""Rendered pixels follow physics, camera motion and live colors through OVStage."""

from __future__ import annotations

import importlib.util
import unittest
from dataclasses import replace

import numpy as np

import robosimstudio as rss

TIER = "medium"
"""Needs an RTX GPU and the optional renderer runtimes."""


@unittest.skipUnless(
    importlib.util.find_spec("ovrtx") and importlib.util.find_spec("ovstage"), "the RTX extra is not installed"
)
class TestRuntimeSceneUpdates(unittest.TestCase):
    def test_physics_camera_and_color_changes_reach_the_next_frame(self):
        from robosimstudio.solvers.render import rt_cores

        available, reason = rt_cores()
        if not available:
            self.skipTest(reason)
        scene = rss.Scene(cell="bench_075", physics=rss.MUJOCO_500HZ)
        cube = scene.add(rss.box(name="cube", half=(0.05,) * 3, at=(0, 0, 0.3), color=(1.0, 0.03, 0.03)))
        sim = scene.build()
        camera = rss.CameraSpec(
            name="test", eye=(1.5, -1.2, 1.5), look_at=(0.6, 0, 0.9), width=320, height=240, fov_y=42
        )
        renderer = sim.make_renderer("rtx", cameras=[camera], channels=["rgb"], live_shapes=cube.shapes, accumulate=8)
        try:

            def frame():
                return renderer.render(sim.state)["test"]["rgb"].astype(np.float32)

            def colored(rgb, channel):
                other = [i for i in range(3) if i != channel]
                return (rgb[..., channel] > 80) & (rgb[..., channel] > 2 * rgb[..., other].max(axis=-1))

            def center(rgb):
                pixels = np.argwhere(colored(rgb, 0))
                self.assertGreater(len(pixels), 25, "the red cube is missing from the camera")
                return pixels.mean(axis=0)

            frame()  # populate and warm the renderer
            initial = center(frame())
            sim.run(150).sync()
            self.assertAlmostEqual(cube.position[2], 0.8, delta=0.005)
            landed = center(frame())
            self.assertGreater(np.linalg.norm(landed - initial), 8, "the rendered cube did not follow gravity")

            renderer.cameras = [replace(camera, look_at=(0.6, 0.25, 0.9))]
            shifted = center(frame())
            self.assertGreater(np.linalg.norm(shifted - landed), 8, "the new camera pose was not rendered")

            colors = sim.model.shape_color.numpy()
            colors[cube.shapes] = (0.03, 0.03, 1.0)
            sim.model.shape_color.assign(colors)
            blue = frame()
            self.assertGreater(np.count_nonzero(colored(blue, 2)), 25, "the live color write was not rendered")
            self.assertLess(np.count_nonzero(colored(blue, 0)), 10, "the old red color remained in the frame")
        finally:
            renderer.close()
            sim.close()


if __name__ == "__main__":
    unittest.main()
