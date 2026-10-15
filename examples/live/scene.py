# SPDX-License-Identifier: Apache-2.0
"""Two physical arms share a workbench; the chat agent writes their cooperation."""

import numpy as np

import robosimstudio as rss
from robosimstudio.core.render.finish import Finish
from robosimstudio.core.render.room import HANGAR_CLOSED, LAB_GRID_MAT
from robosimstudio_studio.routine import grip, reach

CELL = "bench_075"
ORIGIN = np.array(rss.get_cell(CELL).surface_frame[:3])
DOWN = (0.0, 1.0, 0.0, 0.0)
# The arms AND graspable rigid objects share MuJoCo's native finger contacts.
# Cloth, cable and foam retain two-way VBD coupling; no grasp latches or pins.
rigid = rss.backend("mujoco", nconmax=512, njmax=512, enable_multiccd=True)
physics = rss.replace(
    rss.GENERIC,
    backends=(
        ("articulated", rigid),
        ("rigid", rigid),
        ("*", rss.backend("vbd", rigid_body_contact_buffer_size=256, rigid_contact_history=True)),
    ),
    collision=rss.CollisionSpec(contact_matching="sticky"),
)
CAPABILITIES = {
    "arms": "Two YAM robots named left (positive Y) and right (negative Y). Both can reach the centre.",
    "cube": "Red 40 mm cube on MuJoCo: physical friction grasp, lift, carry and place. Start world (0.52, 0.24, 0.773).",
    "teal_cube": "Teal 40 mm cube on MuJoCo. Start world (0.64, -0.25, 0.773).",
    "exchange": "Shared table handoff zone centred world (0.52, 0.0, 0.75). Place down and withdraw before the other arm approaches.",
    "targets": "Right amber pad centre (0.52, -0.26, 0.75); left silver pad centre (0.64, 0.26, 0.75).",
    "cloth": "Contact-only VBD cloth; no reliable pinch under proxy coupling.",
    "cable": "Contact-only VBD rope; bend or sweep, no latch.",
    "soft_block": "VBD foam; contact deformation, limited compression.",
    "helpers": "drive(sim,session,motion()) drains BOTH arms and steps once per yield. together(*generators) moves arms concurrently. reach's default tolerance is 6 mm; physical checks must use measured object positions. DOWN points the gripper toward the table.",
}
scene = rss.Scene(cell=CELL, name="cooperative_workbench", physics=physics, render=HANGAR_CLOSED)
scene.add(rss.robot("yam_bimanual"))
for side, y in (("left", 0.31), ("right", -0.31)):
    scene.add(
        rss.slab(
            name=f"{side}_pedestal",
            half=(0.07, 0.09, 0.386),
            at=(-0.3475, y, -0.364),
            collide=False,
            color=(0.28, 0.30, 0.32),
            finish=Finish(roughness=0.36, metallic=0.85),
        )
    )
scene.add(
    rss.CameraSpec(name="demo", eye=(1.65, 1.10, 1.75), look_at=(0.48, 0.0, 0.89), width=1600, height=1000, fov_y=43)
)
scene.add(
    rss.slab(
        name="work_mat",
        half=(0.293, 0.645, 0.00025),
        at=(0, 0, 0.0003),
        collide=False,
        color=(0.30, 0.32, 0.34),
        finish=LAB_GRID_MAT,
    )
)
scene.add(
    rss.box(
        name="cube",
        half=(0.02,) * 3,
        at=(-0.08, 0.24, 0.023),
        mass=0.05,
        friction=0.9,
        color=(0.70, 0.075, 0.045),
        finish=Finish(roughness=0.3),
    )
)
scene.add(
    rss.box(
        name="teal_cube",
        half=(0.02,) * 3,
        at=(0.04, -0.25, 0.023),
        mass=0.05,
        friction=0.9,
        color=(0.025, 0.38, 0.34),
        finish=Finish(roughness=0.3),
    )
)
for name, at, color in (
    ("exchange", (-0.08, 0.0, 0.0006), (0.25, 0.31, 0.36)),
    ("right_target", (-0.08, -0.26, 0.0006), (0.80, 0.47, 0.10)),
    ("left_target", (0.04, 0.26, 0.0006), (0.62, 0.67, 0.70)),
):
    scene.add(
        rss.slab(
            name=name,
            half=(0.055, 0.055, 0.00025),
            at=at,
            collide=False,
            color=color,
            finish=Finish(roughness=0.38, metallic=0.65),
        )
    )
scene.add(
    rss.sheet(
        name="cloth",
        size=(0.13, 0.16),
        cells=(12, 14),
        at=(0.13, 0.32, 0.012),
        material=rss.materials.TERRY_TOWELLING,
        color=(0.55, 0.61, 0.63),
        settle_frames=30,
    )
)
scene.add(
    rss.rod(
        name="cable",
        length=0.27,
        radius=0.007,
        segments=16,
        at=(0.19, -0.25, 0.016),
        along=(0, 1, 0),
        color=(0.78, 0.31, 0.055),
        allow_grasp_latch=False,
        filter_fingertip_scenery=False,
        scene_friction=False,
    )
)
scene.add(
    rss.SoftSpec(
        name="soft_block",
        size=(0.075, 0.06, 0.04),
        cells=(5, 4, 3),
        pose=(0.12, -0.48, 0.006),
        color=(0.32, 0.21, 0.37),
        settle_frames=30,
    )
)


def together(*motions):
    """Advance independent arm motions in one shared control step."""
    active = list(motions)
    while active:
        active = [motion for motion in active if next(motion, False) is not False]
        if active:
            yield


def drive(sim, session, motion):
    """Execute a model-authored generator through both arms' normal IK controls."""
    paused = session.paused
    session.set_paused(False)
    try:
        for _ in motion:
            for panel in session.arms.values():
                panel.drain()
            sim.step_teleop()
            yield
    finally:
        session.set_paused(paused)


M = {}


def _cmd(ret):
    """Advance a helper motion if it is a generator; otherwise hold one control step."""
    if hasattr(ret, "__iter__") or hasattr(ret, "__next__"):
        yield from ret
    else:
        yield


def _cube(sim):
    return np.asarray(sim["cube"].position, dtype=float)


def _handoff_motion(sim, session):
    M.clear()
    down = DOWN
    cx, cy = 0.52, 0.24  # red cube start (world)
    ex, ey = 0.52, 0.0  # exchange pad centre (world)
    ax, ay = 0.52, -0.26  # amber pad centre (world)
    grasp_z, carry_z = 0.782, 0.86

    # ---- left arm: grasp, lift, carry to exchange, set down, clear ----
    yield from _cmd(grip(session, "left", 0.0))
    yield from _cmd(reach(session, "left", (cx, cy, 0.90), down, speed=0.25))
    yield from _cmd(reach(session, "left", (cx, cy, grasp_z), down, speed=0.06))
    yield from _cmd(grip(session, "left", 1.0))
    yield from _cmd(reach(session, "left", (cx, cy, grasp_z + 0.008), down, speed=0.012))
    p = _cube(sim)
    M["z_before_left_lift_mm"] = float(p[2] * 1000.0)
    yield from _cmd(reach(session, "left", (cx, cy, carry_z), down, speed=0.12))
    p = _cube(sim)
    M["left_lift_mm"] = float(p[2] * 1000.0 - M["z_before_left_lift_mm"])
    yield from _cmd(reach(session, "left", (ex, ey, carry_z), down, speed=0.20))
    yield from _cmd(reach(session, "left", (ex, ey, grasp_z), down, speed=0.05))
    yield from _cmd(grip(session, "left", 0.0))
    yield from _cmd(reach(session, "left", (ex, ey, grasp_z + 0.016), down, speed=0.015))
    p = _cube(sim)
    M["exchange_error_mm"] = float(np.hypot(p[0] - ex, p[1] - ey) * 1000.0)
    M["exchange_cube"] = [float(v) for v in p]
    yield from _cmd(reach(session, "left", (ex, ey, 0.88), down, speed=0.10))
    yield from _cmd(reach(session, "left", (0.58, 0.18, 0.88), down, speed=0.20))
    yield from _cmd(reach(session, "left", (0.62, 0.31, 0.92), down, speed=0.15))

    # ---- right arm: pick from exchange, place on amber pad ----
    yield from _cmd(grip(session, "right", 0.0))
    yield from _cmd(reach(session, "right", (ex, ey, 0.90), down, speed=0.25))
    yield from _cmd(reach(session, "right", (ex, ey, grasp_z), down, speed=0.06))
    yield from _cmd(grip(session, "right", 1.0))
    yield from _cmd(reach(session, "right", (ex, ey, grasp_z + 0.008), down, speed=0.012))
    p = _cube(sim)
    M["z_before_right_lift_mm"] = float(p[2] * 1000.0)
    yield from _cmd(reach(session, "right", (ex, ey, carry_z), down, speed=0.12))
    p = _cube(sim)
    M["right_lift_mm"] = float(p[2] * 1000.0 - M["z_before_right_lift_mm"])
    yield from _cmd(reach(session, "right", (ax, ay, carry_z), down, speed=0.20))
    yield from _cmd(reach(session, "right", (ax, ay, grasp_z), down, speed=0.05))
    yield from _cmd(grip(session, "right", 0.0))
    yield from _cmd(reach(session, "right", (ax, ay, grasp_z + 0.016), down, speed=0.015))
    p = _cube(sim)
    M["release_cube"] = [float(v) for v in p]
    p0 = _cube(sim)
    for _ in range(30):
        yield
    p1 = _cube(sim)
    M["rest_drift_mm"] = float(np.linalg.norm(p1 - p0) * 1000.0)
    M["final_cube"] = [float(v) for v in p1]
    yield from _cmd(reach(session, "right", (ax, ay, 0.88), down, speed=0.10))
    yield from _cmd(reach(session, "right", (0.58, -0.18, 0.88), down, speed=0.20))
    yield from _cmd(reach(session, "right", (0.62, -0.31, 0.92), down, speed=0.15))


def _resolve(*args, **kw):
    """Locate sim and session from however the runner invokes the routine."""
    args = [a for a in args if a is not None]
    sim = kw.get("sim")
    session = kw.get("session")
    if len(args) >= 2:
        if sim is None:
            sim = args[0]
        if session is None:
            session = args[1]
        args = args[2:]
    for a in args + [v for k, v in kw.items() if k not in ("sim", "session")]:
        if a is None:
            continue
        if session is None and hasattr(a, "arms") and hasattr(a, "set_paused"):
            session = a
        elif sim is None and (hasattr(a, "step_teleop") or hasattr(a, "state")):
            sim = a
        elif session is None and hasattr(a, "arms"):
            session = a
        elif sim is None:
            sim = a
    if session is None:
        session = globals().get("session")
    if sim is None:
        sim = globals().get("sim")
    if sim is None and session is not None:
        sim = getattr(session, "sim", None) or (session if hasattr(session, "step_teleop") else None)
    if session is None and sim is not None:
        session = getattr(sim, "session", None) or (sim if hasattr(sim, "arms") else None)
    if sim is None or session is None:
        raise TypeError(f"handoff could not resolve sim/session from args={args!r} kw={list(kw)!r}")
    return sim, session


def handoff(sim=None, session=None, *extra, **kw):
    """Left arm hands the red cube over at the exchange pad; right arm finishes on the amber pad."""
    sim, session = _resolve(*[a for a in (sim, session, *extra) if a is not None], **kw)
    yield from drive(sim, session, _handoff_motion(sim, session))


def handoff_outcome(sim):
    p = np.asarray(sim["cube"].position, dtype=float)
    final_error = float(np.hypot(p[0] - 0.52, p[1] + 0.26) * 1000.0)
    left_lift = float(M.get("left_lift_mm", 0.0))
    right_lift = float(M.get("right_lift_mm", 0.0))
    exch = float(M.get("exchange_error_mm", 999.0))
    drift = float(M.get("rest_drift_mm", 999.0))
    on_table = bool(0.74 < float(p[2]) < 0.80)
    passed = bool(
        left_lift > 15.0 and right_lift > 15.0 and exch < 30.0 and final_error < 35.0 and on_table and drift < 2.0
    )
    return {
        "left_lift_mm": left_lift,
        "right_lift_mm": right_lift,
        "exchange_error_mm": exch,
        "final_error_mm": final_error,
        "final_cube": [float(v) for v in p],
        "rest_drift_mm": drift,
        "on_table": on_table,
        "passed": passed,
    }


def _reverse_motion(sim, session):
    M.clear()
    down = DOWN
    tx, ty = 0.64, -0.25  # teal cube start (world)
    ex, ey = 0.52, 0.0  # exchange pad centre (world)
    sx, sy = 0.64, 0.26  # silver pad centre (world)
    grasp_z, carry_z = 0.782, 0.86

    def _teal(sim):
        return np.asarray(sim["teal_cube"].position, dtype=float)

    # ---- right arm: grasp teal cube, lift, carry to exchange, set down, clear ----
    yield from _cmd(grip(session, "right", 0.0))
    yield from _cmd(reach(session, "right", (tx, ty, 0.90), down, speed=0.25))
    yield from _cmd(reach(session, "right", (tx, ty, grasp_z), down, speed=0.06))
    yield from _cmd(grip(session, "right", 1.0))
    yield from _cmd(reach(session, "right", (tx, ty, grasp_z + 0.008), down, speed=0.012))
    p = _teal(sim)
    M["z_before_right_lift_mm"] = float(p[2] * 1000.0)
    yield from _cmd(reach(session, "right", (tx, ty, carry_z), down, speed=0.12))
    p = _teal(sim)
    M["right_lift_mm"] = float(p[2] * 1000.0 - M["z_before_right_lift_mm"])
    yield from _cmd(reach(session, "right", (ex, ey, carry_z), down, speed=0.20))
    yield from _cmd(reach(session, "right", (ex, ey, grasp_z), down, speed=0.05))
    yield from _cmd(grip(session, "right", 0.0))
    yield from _cmd(reach(session, "right", (ex, ey, grasp_z + 0.016), down, speed=0.015))
    p = _teal(sim)
    M["exchange_error_mm"] = float(np.hypot(p[0] - ex, p[1] - ey) * 1000.0)
    M["exchange_cube"] = [float(v) for v in p]
    yield from _cmd(reach(session, "right", (ex, ey, 0.88), down, speed=0.10))
    yield from _cmd(reach(session, "right", (0.58, -0.18, 0.88), down, speed=0.20))
    yield from _cmd(reach(session, "right", (0.62, -0.31, 0.92), down, speed=0.15))

    # ---- left arm: pick from exchange, place on silver pad ----
    yield from _cmd(grip(session, "left", 0.0))
    yield from _cmd(reach(session, "left", (ex, ey, 0.90), down, speed=0.25))
    yield from _cmd(reach(session, "left", (ex, ey, grasp_z), down, speed=0.06))
    yield from _cmd(grip(session, "left", 1.0))
    yield from _cmd(reach(session, "left", (ex, ey, grasp_z + 0.008), down, speed=0.012))
    p = _teal(sim)
    M["z_before_left_lift_mm"] = float(p[2] * 1000.0)
    yield from _cmd(reach(session, "left", (ex, ey, carry_z), down, speed=0.12))
    p = _teal(sim)
    M["left_lift_mm"] = float(p[2] * 1000.0 - M["z_before_left_lift_mm"])
    yield from _cmd(reach(session, "left", (sx, sy, carry_z), down, speed=0.20))
    yield from _cmd(reach(session, "left", (sx, sy, grasp_z), down, speed=0.05))
    yield from _cmd(grip(session, "left", 0.0))
    yield from _cmd(reach(session, "left", (sx, sy, grasp_z + 0.016), down, speed=0.015))
    p = _teal(sim)
    M["release_cube"] = [float(v) for v in p]
    p0 = _teal(sim)
    for _ in range(30):
        yield
    p1 = _teal(sim)
    M["rest_drift_mm"] = float(np.linalg.norm(p1 - p0) * 1000.0)
    M["final_cube"] = [float(v) for v in p1]
    yield from _cmd(reach(session, "left", (sx, sy, 0.88), down, speed=0.10))
    yield from _cmd(reach(session, "left", (0.58, 0.18, 0.88), down, speed=0.20))
    yield from _cmd(reach(session, "left", (0.62, 0.31, 0.92), down, speed=0.15))


def reverse_handoff(sim, session):
    """Right arm hands the teal cube over at the exchange pad; left arm finishes on the silver pad."""
    sim, session = _resolve(sim, session)
    yield from drive(sim, session, _reverse_motion(sim, session))


def reverse_handoff_outcome(sim):
    p = np.asarray(sim["teal_cube"].position, dtype=float)
    final_error = float(np.hypot(p[0] - 0.64, p[1] - 0.26) * 1000.0)
    right_lift = float(M.get("right_lift_mm", 0.0))
    left_lift = float(M.get("left_lift_mm", 0.0))
    exch = float(M.get("exchange_error_mm", 999.0))
    drift = float(M.get("rest_drift_mm", 999.0))
    on_table = bool(0.74 < float(p[2]) < 0.80)
    passed = bool(
        right_lift > 15.0 and left_lift > 15.0 and exch < 30.0 and final_error < 35.0 and on_table and drift < 2.0
    )
    return {
        "right_lift_mm": right_lift,
        "left_lift_mm": left_lift,
        "exchange_error_mm": exch,
        "final_error_mm": final_error,
        "final_cube": [float(v) for v in p],
        "rest_drift_mm": drift,
        "on_table": on_table,
        "passed": passed,
    }


routines = {"handoff": handoff, "reverse_handoff": reverse_handoff}
outcomes = {"handoff": handoff_outcome, "reverse_handoff": reverse_handoff_outcome}
