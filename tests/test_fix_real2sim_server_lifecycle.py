# SPDX-License-Identifier: Apache-2.0
"""The server's teardown, the board's bound, check under a routine, and routine names: no GPU."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from robosimstudio_real2sim_agent import server as server_mod
from robosimstudio_real2sim_agent.page import LiveSession
from robosimstudio_real2sim_agent.server import LiveServer
from robosimstudio_studio.session import StudioConfig, StudioSession

TIER = "fast"


class RebuildPresentation(unittest.TestCase):
    def test_retiring_controls_does_not_reset_client_layout(self) -> None:
        session = LiveSession.__new__(LiveSession)
        session.server = mock.MagicMock()
        root = session.page_root = mock.Mock()
        session.clear_gui()
        root.remove.assert_called_once()
        session.server.gui.reset.assert_not_called()
        self.assertIsNone(session.page_root)

    def test_rebuilding_controls_does_not_reapply_opening_layout(self) -> None:
        session = LiveSession.__new__(LiveSession)
        session.server = mock.MagicMock()
        session._theme_configured = False
        session.config = StudioConfig()
        with mock.patch.object(StudioSession, "configure_gui") as configure:
            session.configure_gui()
            session.configure_gui()
        configure.assert_called_once()
        session.server.gui.configure_theme.assert_called_once()

    def test_rebuild_keeps_camera_preferences_and_sends_a_complete_frame(self) -> None:
        session = LiveSession.__new__(LiveSession)
        session.host = SimpleNamespace(hold=True)
        session.env = object()
        session.config = StudioConfig(live=True, live_frustums=True)
        session.live = SimpleNamespace(live=False, show_frustums=False)
        session.server = mock.MagicMock()
        session.draw = mock.Mock()
        sim = mock.Mock()
        order = []
        session.server.atomic.return_value.__enter__.side_effect = lambda: order.append("begin")
        session.server.atomic.return_value.__exit__.side_effect = lambda *_: order.append("end")
        session.draw.side_effect = lambda: order.append("draw")
        sim.sync.side_effect = lambda: order.append("sync")
        with mock.patch.object(StudioSession, "adopt", side_effect=lambda *a, **kw: order.append("adopt")):
            session.adopt(sim)
        self.assertFalse(session.config.live)
        self.assertFalse(session.config.live_frustums)
        self.assertEqual(order, ["begin", "adopt", "sync", "draw", "end"])
        # An existing user's viewpoint is untouched by scene changes.
        self.assertEqual(session.server.initial_camera.mock_calls, [])

    def test_first_scene_sets_the_opening_view(self) -> None:
        session = LiveSession.__new__(LiveSession)
        session.host = SimpleNamespace(hold=True)
        session.env = session.live = None
        session.config = StudioConfig()
        session.server = mock.MagicMock()
        session.draw = mock.Mock()
        camera = SimpleNamespace(eye=(1.8, 0.6, 1.7), look_at=(0.35, 0, 1.02), fov_y=42)
        sim = mock.Mock(cameras=[camera])
        with mock.patch.object(StudioSession, "adopt"):
            session.adopt(sim)
        self.assertEqual(session.server.initial_camera.position, camera.eye)
        self.assertEqual(session.server.initial_camera.look_at, camera.look_at)
        self.assertAlmostEqual(session.server.initial_camera.fov, 0.7330382858)


def _bare_server() -> LiveServer:
    live = LiveServer.__new__(LiveServer)
    live.path = Path("/nonexistent/scene.py")
    live.script = []
    live.recorder = None
    live.run_budget = 0
    live.last_error = ""
    live.board = {}
    live.ns = {}
    live.session = mock.MagicMock()
    live.session.paused = False
    return live


class ServeFinishesTheTake(unittest.TestCase):
    def test_quit_mid_take_finishes_the_recorder(self) -> None:
        live = _bare_server()
        live.open_page = lambda: None
        live.socket = mock.MagicMock()
        live.bus = mock.MagicMock()
        live.config = mock.MagicMock(share=False)
        live.url, live.sock_path, live.share_url = "http://x", Path("/tmp/x.sock"), None
        live.session.viewer.is_running.return_value = False
        live.script = [iter(())]
        live.recorder = object()
        finished = []
        live._finish_recorder = finished.append
        with mock.patch.object(server_mod.signal, "signal"):
            live.serve()
        self.assertEqual(finished, ["the server stopped"])
        self.assertEqual(live.script, [])
        live.session.close.assert_called_once()

    def test_a_failing_finish_does_not_skip_the_teardown(self) -> None:
        live = _bare_server()
        live.open_page = lambda: None
        live.socket = mock.MagicMock()
        live.bus = mock.MagicMock()
        live.config = mock.MagicMock(share=False)
        live.url, live.sock_path, live.share_url = "http://x", Path("/tmp/x.sock"), None
        live.session.viewer.is_running.return_value = False
        live.stop_scripts = mock.MagicMock(side_effect=RuntimeError("boom"))
        with mock.patch.object(server_mod.signal, "signal"):
            live.serve()
        live.socket.close.assert_called_once()
        live.session.close.assert_called_once()


class BoardIsBounded(unittest.TestCase):
    def test_board_keeps_the_newest(self) -> None:
        live = _bare_server()
        image = np.zeros((2, 2, 3), np.uint8)
        with mock.patch.object(server_mod.stills, "load_reference", return_value=(image, "x")):
            for i in range(server_mod.BOARD_LIMIT + 5):
                live.cmd_show("p.png", label=f"compare {i}")
        self.assertEqual(len(live.board), server_mod.BOARD_LIMIT)
        self.assertEqual(next(iter(live.board)), "compare 5")
        live.session.clear.assert_any_call("compare 0")

    def test_a_label_shown_again_is_the_newest(self) -> None:
        live = _bare_server()
        image = np.zeros((2, 2, 3), np.uint8)
        with mock.patch.object(server_mod.stills, "load_reference", return_value=(image, "x")):
            live.cmd_show("p.png", label="a")
            live.cmd_show("p.png", label="b")
            live.cmd_show("p.png", label="a")
        self.assertEqual(list(live.board), ["b", "a"])


class CheckUnderARoutine(unittest.TestCase):
    def test_check_is_refused_while_a_routine_plays(self) -> None:
        live = _bare_server()
        live.script = [iter(())]
        with mock.patch.object(server_mod.checks, "gate") as gate:
            reply = live.cmd_check()
        self.assertFalse(reply["ok"])
        self.assertIn("routine is playing", reply["error"])
        gate.assert_not_called()
        self.assertEqual(len(live.script), 1)


class RoutineNames(unittest.TestCase):
    def test_a_non_str_key_plays(self) -> None:
        live = _bare_server()
        live.sim = object()

        def fn(sim):
            yield

        live.ns = {"routines": {1: fn}}
        live.start_script = mock.MagicMock()
        self.assertEqual(live.routine_names(), ["1"])
        reply = live.cmd_play("1")
        self.assertTrue(reply["ok"], reply)
        live.start_script.assert_called_once()

    def test_a_routines_key_named_routine_plays(self) -> None:
        live = _bare_server()
        live.sim = object()

        def fn(sim):
            yield

        live.ns = {"routines": {"routine": fn}}
        live.start_script = mock.MagicMock()
        reply = live.cmd_play("routine")
        self.assertTrue(reply["ok"], reply)


if __name__ == "__main__":
    unittest.main()
