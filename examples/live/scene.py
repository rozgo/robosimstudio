# SPDX-License-Identifier: Apache-2.0
"""One shared table: rigid cube, cloth, cable and soft block for the chat agent."""

import numpy as np

import robosimstudio as rss
from robosimstudio_studio.routine import grip, hand, reach, wait

CELL = "bench_075"
ORIGIN = np.array(rss.get_cell(CELL).surface_frame[:3])
TARGET_Y = -0.18
TARGET_COLOR = (0.93, 0.60, 0.08)
PUSH_SPEED = 0.04
DROP_SPEED = 0.07
BENCH_EDGE_X = 0.3025  # world X of the bench's -X edge (0.6 - 0.2975)
DIRECTION = 1 if TARGET_Y > 0 else -1
CUBE_START = (-0.15, 0.0, 0.026)
TARGET = np.array([-0.15, TARGET_Y, 0.0]) + ORIGIN
DOWN = (0.0, 1.0, 0.0, 0.0)
settle_positions = []
drop_positions = []
poke_data = {}
cable_data = {}

# Two Newton solver islands: the articulated arm on MuJoCo; free bodies,
# cable capsules, cloth and soft particles together on VBD. The proxy coupler
# exchanges forces. This supports pushing; it is not a cloth-pinch setup.
physics = rss.replace(
    rss.GENERIC,
    backends=(
        ("articulated", rss.backend("mujoco")),
        ("*", rss.backend("vbd", rigid_body_contact_buffer_size=256, rigid_contact_history=True)),
    ),
    collision=rss.CollisionSpec(contact_matching="sticky"),
)
CAPABILITIES = {
    "cube": "Red rigid cube: push with robot contact; target pad remains available.",
    "cloth": "Blue cloth: push or drape; proxy coupling does not support a reliable friction pinch.",
    "cable": "Orange capsule-chain rope: bend or sweep by contact; grasp latches are disabled.",
    "soft_block": "Purple VBD tetrahedral block: deforms on contact; strong compression is limited.",
}
scene = rss.Scene(cell=CELL, name="shared_table", physics=physics)
scene.add(rss.robot("panda"))
scene.add(
    rss.CameraSpec(
        name="demo",
        eye=(1.80, 0.75, 1.85),
        look_at=(0.42, 0.0, 0.94),
        width=1280,
        height=720,
        fov_y=42,
    )
)

scene.add(
    rss.box(
        name="cube",
        half=(0.025,) * 3,
        at=CUBE_START,
        mass=0.08,
        friction=0.45,
        color=(0.88, 0.12, 0.12),
    )
)
scene.add(
    rss.slab(
        name="target",
        half=(0.065, 0.065, 0.0003),
        at=(-0.15, TARGET_Y, 0.0004),
        collide=False,
        color=TARGET_COLOR,
    )
)
scene.add(
    rss.sheet(
        name="cloth",
        size=(0.22, 0.24),
        cells=(16, 18),
        at=(-0.22, 0.17, 0.012),
        material=rss.materials.TERRY_TOWELLING,
        color=(0.10, 0.55, 0.86),
        settle_frames=30,
    )
)
scene.add(
    rss.rod(
        name="cable",
        length=0.34,
        radius=0.009,
        segments=16,
        at=(0.08, -0.45, 0.016),
        along=(0.0, 1.0, 0.0),
        color=(1.0, 0.38, 0.06),
        allow_grasp_latch=False,
        filter_fingertip_scenery=False,
        scene_friction=False,  # keep the other objects' own friction values
    )
)
scene.add(
    rss.SoftSpec(
        name="soft_block",
        size=(0.09, 0.07, 0.045),
        cells=(6, 5, 3),
        pose=(0.14, 0.08, 0.006),
        color=(0.60, 0.24, 0.78),
        settle_frames=30,
    )
)


def outcome(sim):
    """Measure the cube after the physical push, in metres and metres/second."""
    position = np.array(sim["cube"].position)
    error = float(np.linalg.norm(position[:2] - TARGET[:2]))
    speed = float(np.linalg.norm(sim["cube"].velocity[:3]))
    # Contact solvers can report a nonzero instantaneous velocity while a
    # resting cube stays within a fraction of a millimetre. Measure the actual
    # positions throughout the final 0.6 s, as the live page's rest gate does.
    drift = float(np.linalg.norm(np.ptp(settle_positions, axis=0))) if len(settle_positions) >= 25 else None
    return {
        "cube": position.tolist(),
        "target": TARGET.tolist(),
        "error_mm": round(1000 * error, 1),
        "speed_mm_s": round(1000 * speed, 1),
        "drift_mm": round(1000 * drift, 3) if drift is not None else None,
        "passed": error < 0.035 and drift is not None and drift < 0.002,
    }


def drop_outcome(sim):
    """Measure the cube after the drop attempt, in metres above the ground."""
    position = np.array(sim["cube"].position)
    speed = float(np.linalg.norm(sim["cube"].velocity[:3]))
    off_bench = bool(abs(position[0] - 0.6) > 0.2975 or abs(position[1]) > 0.65)
    fell = bool(position[2] < ORIGIN[2] - 0.10 and off_bench)
    drift = float(np.linalg.norm(np.ptp(drop_positions, axis=0))) if len(drop_positions) >= 25 else None
    return {
        "cube": [float(v) for v in position],
        "height_mm_above_ground": float(round(float(1000 * position[2]), 1)),
        "drop_mm_below_surface": float(round(float(1000 * (ORIGIN[2] - position[2])), 1)),
        "off_bench": bool(off_bench),
        "speed_mm_s": float(round(1000 * speed, 1)),
        "drift_mm": float(round(1000 * drift, 3)) if drift is not None else None,
        "passed": bool(fell),
    }


def _particle_points(sim, name):
    """Return the (N, 3) particle positions of a cloth or soft body."""
    q = sim[name].particle_q
    if hasattr(q, "detach"):
        q = q.detach().cpu().numpy()
    arr = np.asarray(q)
    return np.ascontiguousarray(arr.reshape(-1, arr.shape[-1])[:, :3], dtype=float)


def _cable_points(sim):
    """Return all capsule centres of the cable."""
    q = sim.state.body_q
    if hasattr(q, "numpy"):
        q = q.numpy()
    arr = np.asarray(q)
    bodies = sim["cable"].bodies
    rows = arr[bodies] if isinstance(bodies, slice) else arr[[int(b) for b in bodies]]
    return np.ascontiguousarray(rows[:, :3], dtype=float)


def _on_table(pts):
    """True when every point stays over the bench top (footprint 0.2975 x 0.65)."""
    pts = np.asarray(pts, dtype=float)
    return bool(
        np.all(np.abs(pts[:, 0] - 0.6) <= 0.3025) and np.all(np.abs(pts[:, 1]) <= 0.655) and np.all(pts[:, 2] >= 0.745)
    )


def poke_outcome(sim):
    """Measure the soft block's vertical compression and recovery, in mm."""
    soft = _particle_points(sim, "soft_block")
    start = poke_data.get("initial")
    if start is None or start.shape != soft.shape:
        return {"passed": False, "error": "Run poke to record the starting particle positions."}
    pressed = poke_data.get("pressed") or []
    on_table = {
        "cube": _on_table(np.array(sim["cube"].position).reshape(1, 3)),
        "cloth": _on_table(_particle_points(sim, "cloth")),
        "cable": _on_table(_cable_points(sim)),
        "soft_block": _on_table(soft),
    }
    z0 = float(start[:, 2].max())
    h0 = float(start[:, 2].max() - start[:, 2].min())
    depth = max(0.0, z0 - min(float(p[:, 2].max()) for p in pressed)) if pressed else 0.0
    h1 = float(soft[:, 2].max() - soft[:, 2].min())
    residual = max(0.0, z0 - float(soft[:, 2].max()))
    slide = float(np.linalg.norm(soft[:, :2].mean(axis=0) - start[:, :2].mean(axis=0)))
    all_ok = all(on_table.values())
    return {
        "soft_block_centroid": [float(v) for v in soft.mean(axis=0)],
        "height_before_mm": float(round(1000 * h0, 1)),
        "peak_compression_mm": float(round(1000 * depth, 1)),
        "height_after_mm": float(round(1000 * h1, 1)),
        "residual_indent_mm": float(round(1000 * residual, 1)),
        "slide_mm": float(round(1000 * slide, 1)),
        "on_table": {k: bool(v) for k, v in on_table.items()},
        "all_on_table": bool(all_ok),
        "passed": bool(depth > 0.002 and all_ok and slide < 0.025),
    }


def push(sim, session):
    """Move the closed hand behind the cube, push it, and withdraw."""
    arm = next(iter(session.arms))
    panel = session.arms[arm]
    side = np.array([0.0, float(DIRECTION), 0.0])
    settle_positions.clear()

    def motion():
        yield from wait(session, 0.4)
        start = np.array(sim["cube"].position)
        behind = start - side * 0.095
        behind[2] = ORIGIN[2] + 0.030
        above = behind + np.array([0.0, 0.0, 0.15])
        yield from grip(session, arm, 1.0, 0.3)
        yield from reach(session, arm, above, DOWN, speed=0.20)
        yield from reach(session, arm, behind, DOWN, speed=0.10)
        end = TARGET - side * 0.043
        end[2] = behind[2]
        yield from reach(session, arm, end, DOWN, speed=PUSH_SPEED)
        yield from reach(session, arm, end - side * 0.04, DOWN, speed=0.12)
        yield from reach(session, arm, hand(session, arm)[0] + np.array([0, 0, 0.15]), speed=0.18)
        yield from reach(session, arm, *panel.ik.home_pose, speed=0.20)
        for _ in wait(session, 0.6):
            yield
            settle_positions.append(np.array(sim["cube"].position))

    # Live-page generators own their control steps; the normal page's arm
    # helpers only yield, so drain their commands and take one physics step.
    was_paused = session.paused
    session.set_paused(False)
    try:
        for _ in motion():
            panel.drain()
            sim.step_teleop()
            yield
    finally:
        session.set_paused(was_paused)
    result = outcome(sim)
    print(f"Push result: {result}", flush=True)
    if not result["passed"]:
        raise RuntimeError(f"Cube missed its target: {result}")


def drop(sim, session):
    """Push the cube along -X until it clears the bench edge and falls off."""
    arm = next(iter(session.arms))
    panel = session.arms[arm]
    side = np.array([-1.0, 0.0, 0.0])
    drop_positions.clear()

    def motion():
        yield from wait(session, 0.4)
        start = np.array(sim["cube"].position)
        behind = start - side * 0.095
        behind[2] = ORIGIN[2] + 0.030
        above = behind + np.array([0.0, 0.0, 0.15])
        yield from grip(session, arm, 1.0, 0.3)
        yield from reach(session, arm, above, DOWN, speed=0.20)
        yield from reach(session, arm, behind, DOWN, speed=0.10)
        # Sweep the hand well past the bench edge so the cube is driven clear
        # of the lip and tips off instead of stopping on it.
        end = np.array([BENCH_EDGE_X - 0.075, start[1], behind[2]])
        yield from reach(session, arm, end, DOWN, speed=DROP_SPEED)
        yield from reach(session, arm, end + side * 0.05, DOWN, speed=0.12)
        yield from reach(session, arm, hand(session, arm)[0] + np.array([0, 0, 0.20]), speed=0.18)
        yield from reach(session, arm, *panel.ik.home_pose, speed=0.20)
        for _ in wait(session, 0.8):
            yield
            drop_positions.append(np.array(sim["cube"].position))

    was_paused = session.paused
    session.set_paused(False)
    try:
        for _ in motion():
            panel.drain()
            sim.step_teleop()
            yield
    finally:
        session.set_paused(was_paused)
    result = drop_outcome(sim)
    print(f"Drop result: {result}", flush=True)


def poke(sim, session):
    """Press the closed fingertip into the soft block's top face, hold, withdraw."""
    arm = next(iter(session.arms))
    panel = session.arms[arm]
    pressed = []
    poke_data.clear()
    poke_data["pressed"] = pressed

    def motion():
        yield from wait(session, 0.4)
        initial = _particle_points(sim, "soft_block").copy()
        poke_data["initial"] = initial
        top = float(initial[:, 2].max())
        centre = initial.mean(axis=0)
        above = np.array([centre[0], centre[1], ORIGIN[2] + 0.16])
        press = np.array([centre[0], centre[1], top - 0.005])
        yield from grip(session, arm, 1.0, 0.3)
        yield from reach(session, arm, above, DOWN, speed=0.20)
        yield from reach(session, arm, press, DOWN, speed=0.02)
        for _ in wait(session, 0.5):
            yield
            pressed.append(_particle_points(sim, "soft_block").copy())
        yield from reach(session, arm, above, DOWN, speed=0.08)
        yield from reach(session, arm, *panel.ik.home_pose, speed=0.20)
        for _ in wait(session, 0.8):
            yield

    was_paused = session.paused
    session.set_paused(False)
    try:
        for _ in motion():
            panel.drain()
            sim.step_teleop()
            yield
    finally:
        session.set_paused(was_paused)
    result = poke_outcome(sim)
    print(f"Poke result: {result}", flush=True)


def nudge_outcome(sim):
    """Measure the cable's motion and bending from robot contact, in mm."""
    final = _cable_points(sim)
    start = cable_data.get("initial")
    if start is None or np.shape(start) != np.shape(final):
        return {"passed": False, "error": "Run nudge_cable to record the starting capsule positions."}
    disp = np.linalg.norm(final - start, axis=1)
    lateral = np.abs(final[:, 0] - start[:, 0])
    chord0 = float(np.linalg.norm(start[-1] - start[0]))
    chord1 = float(np.linalg.norm(final[-1] - final[0]))
    on_table = {
        "cube": _on_table(np.array(sim["cube"].position).reshape(1, 3)),
        "cloth": _on_table(_particle_points(sim, "cloth")),
        "cable": _on_table(_cable_points(sim)),
        "soft_block": _on_table(_particle_points(sim, "soft_block")),
    }
    all_ok = bool(all(on_table.values()))
    return {
        "cable_centroid": [float(v) for v in final.mean(axis=0)],
        "max_capsule_move_mm": float(round(1000.0 * float(disp.max()), 1)),
        "mean_capsule_move_mm": float(round(1000.0 * float(disp.mean()), 1)),
        "max_lateral_deflection_mm": float(round(1000.0 * float(lateral.max()), 1)),
        "chord_shortening_mm": float(round(1000.0 * (chord0 - chord1), 1)),
        "on_table": {k: bool(v) for k, v in on_table.items()},
        "all_on_table": all_ok,
        "passed": bool(float(disp.max()) > 0.005 and all_ok),
    }


def nudge_cable(sim, session):
    """Sweep the closed fist across the cable to bend it sideways, then withdraw."""
    arm = next(iter(session.arms))
    panel = session.arms[arm]
    cable_data.clear()

    def motion():
        yield from wait(session, 0.4)
        cable_data["initial"] = _cable_points(sim).copy()
        pts = cable_data["initial"]
        mid_y = float(pts[:, 1].mean())
        above = np.array([0.705, mid_y, ORIGIN[2] + 0.16])
        low = np.array([0.705, mid_y, ORIGIN[2] + 0.025])
        end = np.array([0.635, mid_y, ORIGIN[2] + 0.025])
        yield from grip(session, arm, 1.0, 0.3)
        yield from reach(session, arm, above, DOWN, speed=0.20)
        yield from reach(session, arm, low, DOWN, speed=0.05)
        yield from reach(session, arm, end, DOWN, speed=0.03)
        yield from reach(session, arm, end + np.array([0, 0, 0.15]), speed=0.10)
        yield from reach(session, arm, *panel.ik.home_pose, speed=0.20)
        for _ in wait(session, 0.6):
            yield

    was_paused = session.paused
    session.set_paused(False)
    try:
        for _ in motion():
            panel.drain()
            sim.step_teleop()
            yield
    finally:
        session.set_paused(was_paused)
    result = nudge_outcome(sim)
    print(f"Cable nudge result: {result}", flush=True)


outcomes = {"push": outcome, "drop": drop_outcome, "poke": poke_outcome, "nudge_cable": nudge_outcome}
routines = {"push": push, "drop": drop, "poke": poke, "nudge_cable": nudge_cable}

if __name__ == "__main__":
    rss.view(scene.build(control_mode="joint_position"), routines=routines)
