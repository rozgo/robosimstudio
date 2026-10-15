# SPDX-License-Identifier: Apache-2.0
"""The studio's page with a ``Live`` folder on top, held still between commands.

The folder is built *first*, because viser sorts the sidebar by creation
order and the controls are rebuilt on every ``adopt`` -- so it is rebuilt
here, on every ``build_gui``, and the server
rewrites its readouts straight after. Markdown rather than text boxes: a text
box is one narrow line and truncates, and these are the two readouts the page
is for -- what is loaded and when, and the traceback when a rebuild did not
land. Whether a rebuild lands running or paused is the server's ``hold``; the
Pause checkbox on the page is the human's override either way, and the
studio's own ``set_paused`` keeps the two truthful. While a routine plays the
page's own stepping is held regardless (:meth:`LiveSession.steps_held`), and
both Reset buttons -- this folder's and the Simulation folder's -- go through
the one :meth:`LiveSession.reset_now`.

The controls share one headerless owner so replacing them preserves the
client's drawer layout. Scene replacement is sent with its first posed frame
in one atomic batch; camera display preferences survive the replacement.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from robosimstudio.sim.draw import title_page
from robosimstudio_studio.session import StudioConfig, StudioSession

if TYPE_CHECKING:
    import numpy as np

    from .server import LiveServer

__all__ = ["LiveSession"]


class LiveSession(StudioSession):
    """The studio's page over a built ``Sim``, with the server's folder at the top of its sidebar."""

    def __init__(self, sim: Any, *, config: StudioConfig, host: LiveServer) -> None:
        self.host = host
        self.host_folder: Any = None
        self.host_widgets: dict[str, Any] = {}
        self.page_root: Any = None
        self._theme_configured = False
        super().__init__(sim, config=config)

    def adopt(self, sim: Any, *, label: str = "", episode: int = 0) -> None:
        # Running, so the page can be dragged and played with the moment it
        # lands; `--hold` lands it paused instead, and `ctl run N` steps a
        # held page.
        self.paused = self.host.hold
        # These are viewing preferences, not properties of a new scene. A
        # hidden frustum must not flash back on whenever an agent saves a file.
        if self.live is not None:
            self.config = replace(self.config, live=self.live.live, live_frustums=self.live.show_frustums)
        view = getattr(self, "render_view", None)
        locked = getattr(view, "locked", "")
        if view is not None:
            self.config = replace(
                self.config,
                world_model=view.mode == "rtx",
                render_view_scale=view.scale,
                render_view_passes=view.passes,
                render_view_clean=view.clean,
            )
        # Set the opening view before new clients connect. Never move an
        # existing client's camera when the scene under it is replaced.
        if self.env is None and sim.cameras:
            camera = sim.cameras[0]
            self.server.initial_camera.position = camera.eye
            self.server.initial_camera.look_at = camera.look_at
            self.server.initial_camera.fov = math.radians(camera.fov_y)
        # The build already happened with the old view intact. Buffer teardown,
        # new geometry, lighting, controls and the first posed frame together,
        # so clients do not see the empty scene between them.
        with self.server.atomic():
            super().adopt(sim, label=label, episode=episode)
            sim.sync()
            self.draw()
            view = getattr(self, "render_view", None)
            if view is not None:
                view.locked = locked if any(c.name == locked for c in sim.cameras) else ""
                if view.require_photoreal:
                    view._hide_model()

    def steps_held(self) -> bool:
        """While a routine plays it steps the scene itself; the studio's own step would double it."""
        return bool(self.host.script)

    def reset_now(self) -> bool:
        """A reset ends whatever routine is playing and the take filming it, then resets."""
        # Before the reset: a take's last frame is read off the scene as it stands.
        self.host.stop_scripts("reset")
        self.host.resets += 1
        return super().reset_now()

    def clear_gui(self) -> None:
        # Removing our controls leaves each client's drawer width, position
        # and collapsed state intact. gui.reset() resets all of those too.
        if self.page_root is not None:
            self.page_root.remove()
            self.page_root = None

    def configure_gui(self) -> None:
        if not self._theme_configured:
            super().configure_gui()
            self.server.gui.configure_theme(
                control_layout="floating", dark_mode=self.config.dark_mode, show_logo=False, show_share_button=False
            )
            self._theme_configured = True

    def build_gui(self) -> None:
        # A headerless owner lets us retire controls without resetting the
        # client's layout. Every panel, including the Live folder, lives here.
        self.page_root = self.server.gui.add_folder(None)
        with self.page_root:
            self._build_live_gui()

    def _build_live_gui(self) -> None:
        gui = self.server.gui
        gui.set_panel_label("Simulation")
        title_page(self.server, "Live simulation")
        self.host_folder = gui.add_folder("Live")
        self.buttons: dict[str, Any] = {}
        self.board: dict[str, Any] = {}
        with self.host_folder:
            self.host_widgets = {
                "scene": gui.add_markdown(""),
                "sim": gui.add_markdown(""),
                "error": gui.add_markdown(""),
            }
            self._build_buttons()
            if self.host.ref_image is not None:
                self.host_widgets["ref"] = gui.add_image(self.host.ref_image, label=self.host.ref_label)
            for label, image in self.host.board.items():
                self.board[label] = gui.add_image(image, label=label)
        super().build_gui()
        self.host.refresh_live(force=True)

    def _build_buttons(self) -> None:
        """One button per routine the scene file binds, and Reset beside them; the main thread acts on a click."""
        gui = self.server.gui
        names = list(self.host.routine_names())
        if not names:
            return
        for name in names:
            button = gui.add_button(f"Run {name}")
            button.on_click(lambda _event, name=name: self.host.clicks.put(("play", name)))
            self.buttons[name] = button
        reset = gui.add_button("Reset")
        # The same request the Simulation folder's Reset makes.
        reset.on_click(lambda _event: setattr(self, "reset_requested", True))
        self.buttons["__reset__"] = reset

    def show_reference(self, image: np.ndarray, label: str) -> None:
        """Put a picture in the ``Live`` folder, replacing the one there."""
        self.hide_reference()
        with self.host_folder:
            self.host_widgets["ref"] = self.server.gui.add_image(image, label=label)

    def hide_reference(self) -> None:
        old = self.host_widgets.pop("ref", None)
        if old is not None:
            old.remove()

    def show(self, label: str, image: np.ndarray) -> None:
        """Put a picture on the board under ``label``, replacing the one with that label."""
        old = self.board.pop(label, None)
        if old is not None:
            old.remove()
        with self.host_folder:
            self.board[label] = self.server.gui.add_image(image, label=label)

    def clear(self, label: str = "") -> None:
        """Take one picture off the board, or all of them."""
        for key in [label] if label else list(self.board):
            old = self.board.pop(key, None)
            if old is not None:
                old.remove()
