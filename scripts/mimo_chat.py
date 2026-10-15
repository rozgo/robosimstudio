# SPDX-License-Identifier: Apache-2.0
"""A local chat agent that calls MiMo through HF and operates the live scene.

Start examples/live/ctl up --hold, then:
    uv run python scripts/mimo_chat.py --out outputs/mimo-live-session

The browser sends only user messages. Scene mutations are dispatched only from
model tool calls. Events, model responses, code revisions and usage are retained.
The HF credential stays in this process and is never included in the browser API.
"""

from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import hashlib
import json
import os
import threading
import time
import urllib.request
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from robosimstudio_real2sim_agent.wire import request

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "examples/live/scene.py"
SOCKET = ROOT / "examples/live/run/ctl.sock"
MODEL = "XiaomiMiMo/MiMo-V2.6-Pro-RL:novita"
DEFAULT_MAX_OUTPUT_TOKENS = 32768
REQUEST_TIMEOUT_SECONDS = 1800
SYSTEM = """You are the coding agent operating a real running robot simulation.
The operator sends requests through this chat. You alone
choose the scene edits and robot actions. Work through tools; do not claim actions
from text alone. Give concise progress updates before edits or motion and a brief
measured result afterward. Use concise Markdown with short paragraphs or bullets.
Do the requested work without asking for confirmation.

Read the scene and inspect it before changing it. Use the helpers supplied by the
current scene; read its CAPABILITIES and the observed solver ownership. You can
edit any part of scene.py with exact replacements. Batch related
edits into one save. Saving rebuilds the scene and resets the robot and objects; state
this plainly when you do it. Preserve the camera and existing physical outcome checks.
Do not change measurements to manufacture success. Verify the result after action.
Never teleport an object, attach it to the hand, pin particles, enable a grasp
latch, or set any object's final pose directly.
Movement must arise from the robot's contact in the physics engine. If motion fails,
report it and use the evidence to repair the routine, within the request limit.
For a new task, add a new routine and a matching measured outcome; the original
pad-placement check only applies to the original push task. Keep the bench, object
starting pose, dimensions and physics unchanged unless the user asks to edit them.
Keep each new routine focused on one short, measurable physical attempt. Inspect
its result before writing fallback experiments.

Scene API: import robosimstudio as rss. scene = rss.Scene(cell='bench_075',
physics=rss.MUJOCO_500HZ). scene.add(rss.box(name=..., half=(...), at=(...),
mass=..., friction=..., color=(r,g,b))). rss.slab is a fixed pad; collide=False
makes a visual target. Scene at= coordinates are metres relative to the bench
surface, whose world origin is (0.6,0,0.75). sim['cube'].position is world XYZ.
inspect_scene returns the actual bench dimensions, arm names and tool poses.
The current scene may also contain cloth, cable and soft bodies. Read its source
and inspect_scene's objects and physics fields: these are actual live particle
and body positions and the solver manifest. Cloth and soft blocks have particle_q,
not a single rigid position. For cable use sim.state.body_q.numpy()[sim['cable'].bodies,:3]
for all capsule centres; sim['cable'].position is only its first capsule.
Scene CAPABILITIES describes the available contact behavior. Rigid objects in
the same MuJoCo island as the arms can be grasped by physical finger friction.
VBD cloth and cable across proxy coupling support pushing and bending; reliable
cloth pinching is not supported. Do not change solvers to make a task pass.
Choose feasible contact motions; report limitations and actual deformation.

Motion helpers come from robosimstudio_studio.routine.
reach(session, arm, world_xyz, quaternion_xyzw=None, speed=metres_per_second)
smoothly moves the hand using IK; grip(session,arm,1.0) closes it. A generator
must call panel.drain() for each arm, sim.step_teleop(), then yield on every
control step; use the scene's drive helper when provided. Define the entry point
with explicit parameters: def task(sim, session). The runner injects session only
when that parameter is named in the signature. routines={'task':task} creates
the Run command. Clear the task's measurement dictionary at the start of each
attempt so stale results cannot pass a later failed run. Preserve existing
outcome checks and register each new one in outcomes={'task':task_outcome}.
run_routine measures outcomes[name](sim), falling back to outcome(sim). Return
actual positions and task-specific measurements, with a boolean passed field.

You receive simulator state as text, not camera vision. The live browser view is
for the operator; do not claim you saw an image. Tools execute synchronously, including
waiting for rebuilds and routines. Do not poll a tool that already completed.
Do not claim success unless run_routine reports actual measured success.
Keep replies short and do not expose private reasoning. You have at most 10 model
requests per user turn. The backend enforces token and session spending limits.
"""


def observe_objects(sim):
    """Compact live geometry and solver ownership, including objects without bodies."""
    import numpy as np

    state = sim.state
    bodies = state.body_q.numpy()[:, :3]
    particles = state.particle_q.numpy() if state.particle_q is not None else np.empty((0, 3))
    objects = {}
    for entity in sim.scene.entities:
        ref = sim[entity.name]
        if ref.particles.size:
            points, storage = particles[ref.particles], "particles"
        elif ref.bodies.size and getattr(entity, "domain", "") != "articulated":
            points, storage = bodies[ref.bodies], "body_centres"
        else:
            continue
        # Round in float64 so JSON does not expand float32 values such as
        # 0.68 into 0.6800000071525574 in every model request.
        points = np.asarray(points, dtype=np.float64)
        indices = np.unique(np.linspace(0, len(points) - 1, min(12, len(points))).astype(int))
        objects[entity.name] = {
            "domain": getattr(entity, "domain", ""),
            "storage": storage,
            "count": len(points),
            "centroid_world": points.mean(axis=0).round(6).tolist(),
            "bounds_world": [points.min(axis=0).round(6).tolist(), points.max(axis=0).round(6).tolist()],
            "samples_world": [{"index": int(i), "xyz": points[i].round(6).tolist()} for i in indices],
        }
    manifest = sim.pipeline.manifest
    return {
        "objects": objects,
        "physics": {
            "coupler": manifest.coupler,
            "islands": [
                {"name": i.name, "solver": i.backend.solver, "entities": list(i.entities)} for i in manifest.islands
            ],
            "channels": [
                {"source": c.source, "destination": c.destination, "coupling": c.coupling} for c in manifest.channels
            ],
        },
    }


def tool(name, description, properties, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    tool("inspect_scene", "Read actual scene state, configuration and available routines.", {}),
    tool("read_scene", "Read the current scene.py source before editing it.", {}),
    tool(
        "edit_scene",
        "Apply exact replacements to scene.py, rebuild, reset and update the live browser. Each old string must occur exactly once. Returns errors from real execution.",
        {
            "replacements": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {"old": {"type": "string"}, "new": {"type": "string"}},
                    "required": ["old", "new"],
                    "additionalProperties": False,
                },
            },
        },
        ("replacements",),
    ),
    tool(
        "run_routine",
        "Run a scene routine in real physics, wait for completion, and return measured outcome and positions. Does not reset the scene.",
        {
            "name": {"type": "string"},
        },
        ("name",),
    ),
]


class Budget:
    """Reserve before sending; keep unresolved reservations after network failures."""

    def __init__(self, path, prices, limit):
        self.path, self.prices, self.limit = path, prices, limit
        if path.exists():
            raise FileExistsError(f"Refusing to reset an existing usage ledger: {path}")
        self.calls = []
        self.budget_changes = []
        self.save()

    @classmethod
    def resume(cls, path, prices, limit, *, allow_budget_increase=False):
        saved = json.loads(path.read_text())
        increasing = allow_budget_increase and limit > saved["limit"]
        if saved["model"] != MODEL or (saved["limit"] != limit and not increasing):
            raise ValueError("Resume must preserve the ledger's model and spending limit.")
        budget = cls.__new__(cls)
        budget.path, budget.prices, budget.limit = path, prices, limit
        budget.calls = saved["calls"]
        budget.budget_changes = saved.get("budget_changes", [])
        if increasing:
            budget.budget_changes.append(
                {
                    "from": saved["limit"],
                    "to": limit,
                    "at": datetime.now(UTC).isoformat(),
                    "authorization": "explicit CLI increase",
                }
            )
            budget.save()
        # Settled costs and unresolved reservations both survive a restart.
        return budget

    @property
    def spent(self):
        return sum(c.get("estimated_usd", 0) for c in self.calls)

    @property
    def reserved(self):
        return sum(c["reserved_usd"] for c in self.calls if "estimated_usd" not in c)

    def reserve(self, body):
        # Byte count plus template allowance is deliberately conservative. It
        # includes tools and the repeated conversation, not just the last prompt.
        input_allowance = len(json.dumps(body, ensure_ascii=True).encode()) + 4096
        reserve = (input_allowance * self.prices["input"] + body["max_tokens"] * self.prices["output"]) / 1e6
        if self.spent + self.reserved + reserve > self.limit:
            raise RuntimeError("Session budget reached: the next request's allowance would exceed the limit.")
        row = {
            "number": len(self.calls) + 1,
            "started_at": datetime.now(UTC).isoformat(),
            "reserved_usd": reserve,
            "max_output_tokens": body["max_tokens"],
            "prices": dict(self.prices),
        }
        self.calls.append(row)
        self.save()
        return row

    def settle(self, row, usage):
        if not usage or not all(k in usage for k in ("prompt_tokens", "completion_tokens")):
            raise RuntimeError("Provider did not return usage; reservation retained and agent stopped.")
        row["usage"] = usage
        # completion_tokens includes reasoning. Do not add reasoning a second
        # time. Full prompt price is conservative if cache credits later apply.
        row["estimated_usd"] = (
            usage["prompt_tokens"] * self.prices["input"] + usage["completion_tokens"] * self.prices["output"]
        ) / 1e6
        self.save()

    def summary(self):
        return {
            "spent": self.spent,
            "reserved": self.reserved,
            "limit": self.limit,
            "requests": len(self.calls),
            "prices": self.prices,
            "input_tokens": sum(c.get("usage", {}).get("prompt_tokens", 0) for c in self.calls),
            "output_tokens": sum(c.get("usage", {}).get("completion_tokens", 0) for c in self.calls),
        }

    def save(self):
        temp = self.path.with_suffix(".json.pending")
        temp.write_text(
            json.dumps(
                {"model": MODEL, **self.summary(), "calls": self.calls, "budget_changes": self.budget_changes}, indent=2
            )
        )
        temp.replace(self.path)


def apply_replacements(source, replacements):
    for change in replacements:
        old = change["old"]
        if not old or source.count(old) != 1:
            raise ValueError(f"Replacement must match exactly once: {old[:100]!r}")
        source = source.replace(old, change["new"], 1)
    ast.parse(source)
    return source


def restore_messages(events, out):
    """Restore completed public exchanges, never replaying a tool or a truncated call."""
    messages = [{"role": "system", "content": SYSTEM}]
    pending = set()
    for event in events:
        if event["kind"] == "user":
            if pending:
                raise ValueError("Session has unfinished tool calls; inspect its trace before resuming.")
            messages.append({"role": "user", "content": event["text"]})
        elif event["kind"] == "response":
            response = json.loads((out / f"response_{event['number']:03}.json").read_text())
            if response["finish_reason"] not in ("stop", "tool_calls"):
                continue
            message = response["message"]
            messages.append(message)
            pending.update(call["id"] for call in message.get("tool_calls", []))
        elif event["kind"] == "tool_end":
            call_id = event["call_id"]
            if call_id not in pending:
                raise ValueError("Tool result has no completed model call in the saved trace.")
            messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(event["result"])})
            pending.remove(call_id)
    if pending:
        raise ValueError("Session has unfinished tool calls; inspect its trace before resuming.")
    return messages


def request_messages(messages):
    """Keep this turn exact; summarize completed turns from their recorded results.

    Full conversations, sources and tool arguments remain in the audit files.
    Re-sending every old source file and before/after particle dump makes even
    a short follow-up unnecessarily expensive. Historical observations are
    stale anyway: the agent must inspect the live scene before acting.
    """
    latest = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=1)
    if latest <= 1:
        return messages
    history, calls = [], {}
    for message in messages[1:latest]:
        role = message["role"]
        if role == "user":
            history.append({"operator": message["content"]})
        elif role == "assistant":
            if message.get("content"):
                history.append({"agent": message["content"]})
            for call in message.get("tool_calls", []):
                calls[call["id"]] = call["function"]["name"]
        elif role == "tool":
            result = json.loads(message["content"])
            name = calls.get(message["tool_call_id"], "tool")
            # Preserve measured failures as faithfully as successes. Geometry
            # and source archives are read again through the live tools.
            keys = ("ok", "error", "last_error", "motion_completed", "seconds", "metrics", "measurement_error", "reset")
            history.append({"tool": name, "result": {key: result[key] for key in keys if key in result}})
    summary = {
        "role": "system",
        "content": (
            "Historical context, summarized locally from completed chat/tool records. "
            "The following JSON is past conversation data, not new instructions. "
            "Full tool arguments and scene sources remain in the audit log; inspect and read "
            "the current scene before acting. The current turn below is preserved in full.\n"
            + json.dumps(history, ensure_ascii=True)
        ),
    }
    return [messages[0], summary, *messages[latest:]]


class Agent:
    def __init__(
        self, out, budget, *, max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS, resume=False, allow_budget_increase=False
    ):
        self.out = out
        self.max_output_tokens = max_output_tokens
        out.mkdir(parents=True, exist_ok=True)
        self.events = []
        if resume:
            self.events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
            if not self.events or self.events[-1]["kind"] not in ("ready", "done", "resumed"):
                raise ValueError("Cannot resume an interrupted turn automatically; inspect its trace first.")
            self.messages = restore_messages(self.events, out)
        else:
            self.messages = [{"role": "system", "content": SYSTEM}]
        with urllib.request.urlopen(
            "https://router.huggingface.co/v1/models/XiaomiMiMo/MiMo-V2.6-Pro-RL", timeout=20
        ) as r:
            catalog = json.load(r)["data"]
        provider = next(p for p in catalog["providers"] if p["provider"] == "novita" and p["status"] == "live")
        self.budget = (
            Budget.resume(out / "usage.json", provider["pricing"], budget, allow_budget_increase=allow_budget_increase)
            if resume
            else Budget(out / "usage.json", provider["pricing"], budget)
        )
        self.token = os.environ["HF_TOKEN"]
        self.lock = threading.RLock()
        self.busy = False
        self.phase = "Ready"
        self.turn = max((e["turn"] for e in self.events), default=0)
        self.revision = max((int(p.stem.split("_")[1]) for p in out.glob("scene_*_after.py")), default=0)
        self.started = time.monotonic() - (self.events[-1]["seconds"] if self.events else 0)
        self.live_url = self.ctl("status")["url"]
        suffix = f"_resume_{len(self.events):03}" if resume else ""
        (out / f"provider{suffix}.json").write_text(json.dumps(catalog, indent=2))
        if not resume:
            (out / "initial_scene.py").write_text(SCENE.read_text())
        (out / f"agent_config{suffix}.json").write_text(
            json.dumps(
                {"model": MODEL, "system": SYSTEM, "tools": TOOLS, "max_output_tokens": max_output_tokens}, indent=2
            )
        )
        self.emit("resumed" if resume else "ready", model=MODEL, max_output_tokens=max_output_tokens)

    def ctl(self, cmd, **kwargs):
        return request(SOCKET, {"cmd": cmd, **kwargs}, timeout=150)

    def read_value(self, expression):
        # eval's transport returns repr(). NumPy scalars now repr as np.float64
        # calls, which literal_eval correctly rejects. Serialize on the sim
        # thread first, converting scalars/arrays through their public tolist.
        reply = self.ctl(
            "eval",
            code=f"__import__('json').dumps(({expression}), default=lambda value: value.tolist(), allow_nan=False)",
        )
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "Could not read the simulator result."))
        return json.loads(ast.literal_eval(reply["value"]))

    def emit(self, kind, **data):
        with self.lock:
            event = {
                "id": len(self.events),
                "seconds": round(time.monotonic() - self.started, 3),
                "kind": kind,
                "turn": self.turn,
                **data,
            }
            self.events.append(event)
            with (self.out / "events.jsonl").open("a") as f:
                f.write(json.dumps(event) + "\n")
        if kind != "delta":
            print(json.dumps(event)[:1200], flush=True)

    def state(self, after):
        with self.lock:
            return {
                "busy": self.busy,
                "phase": self.phase,
                "turn": self.turn,
                "model": MODEL,
                "max_output_tokens": self.max_output_tokens,
                "live_url": self.live_url,
                "budget": self.budget.summary(),
                "events": copy.deepcopy(self.events[after + 1 :]),
            }

    def submit(self, prompt):
        with self.lock:
            if self.busy:
                raise ValueError("The agent is still working on the previous message.")
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
                raise ValueError("Send a nonempty message of at most 4000 characters.")
            self.busy = True
            self.turn += 1
            self.messages.append({"role": "user", "content": prompt})
            self.emit("user", text=prompt)
        threading.Thread(target=self.run, daemon=True).start()

    def completion(self):
        body = {
            "model": MODEL,
            "messages": request_messages(self.messages),
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_tokens": self.max_output_tokens,
            "temperature": 1.0,
            "top_p": 0.95,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        row = self.budget.reserve(body)
        number = row["number"]
        # Retain the externally visible conversation, but not private reasoning.
        public = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in body["messages"]]
        (self.out / f"request_{number:03}.json").write_text(json.dumps({**body, "messages": public}, indent=2))
        self.phase = "Agent is thinking"
        self.emit("request", number=number, reserved_usd=row["reserved_usd"])
        req = urllib.request.Request(
            "https://router.huggingface.co/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"},
        )
        message = {"role": "assistant", "content": ""}
        calls, usage, finish = {}, None, None
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                row["http_status"] = response.status
                row["request_id"] = response.headers.get("x-request-id")
                for line in response:
                    if time.monotonic() - started > REQUEST_TIMEOUT_SECONDS:
                        raise TimeoutError(f"Model request exceeded {REQUEST_TIMEOUT_SECONDS} seconds.")
                    if not line.startswith(b"data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == b"[DONE]":
                        break
                    chunk = json.loads(raw)
                    if chunk.get("error"):
                        raise RuntimeError(str(chunk["error"]))
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices", []):
                        finish = choice.get("finish_reason") or finish
                        delta = choice.get("delta", {})
                        for key in ("content", "reasoning_content"):
                            if delta.get(key):
                                message[key] = message.get(key, "") + delta[key]
                        if delta.get("content"):
                            self.emit("delta", number=number, text=delta["content"])
                        for call in delta.get("tool_calls", []):
                            current = calls.setdefault(
                                call["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                            )
                            if call.get("id"):
                                current["id"] = call["id"]
                            for key in ("name", "arguments"):
                                current["function"][key] += call.get("function", {}).get(key) or ""
            if calls:
                message["tool_calls"] = [calls[k] for k in sorted(calls)]
            row.update(seconds=round(time.monotonic() - started, 3), finish_reason=finish)
            self.budget.settle(row, usage)
            visible = {k: v for k, v in message.items() if k != "reasoning_content"}
            (self.out / f"response_{number:03}.json").write_text(
                json.dumps({"message": visible, "usage": usage, "finish_reason": finish}, indent=2)
            )
            self.emit("response", number=number, message=visible, usage=usage, estimated_usd=row["estimated_usd"])
            if finish == "length":
                reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                raise RuntimeError(
                    f"Response reached its {self.max_output_tokens:,}-token allowance "
                    f"({reasoning:,} reasoning tokens). This response's tool calls were not executed. "
                    f"Estimated request cost: ${row['estimated_usd']:.6f}. "
                    "Earlier completed actions remain; there is no automatic retry."
                )
            if finish not in ("stop", "tool_calls"):
                raise RuntimeError("Provider stream ended without a completed response; no tool calls were executed.")
            return message
        except Exception as exc:
            row["error"] = str(exc).replace(self.token, "[REDACTED]")
            self.budget.save()
            raise

    def inspect(self):
        status = self.ctl("status")
        result = {
            k: status.get(k)
            for k in ("ok", "names", "positions", "routines", "paused", "playing", "generation", "last_error", "finite")
        }
        config = self.ctl(
            "eval",
            code="{k: (v.tolist() if hasattr(v, 'tolist') else v) for k in ('TARGET_Y','TARGET_COLOR','PUSH_SPEED','TARGET','ORIGIN','DIRECTION','CAPABILITIES') if (v := globals().get(k)) is not None}",
        )
        result["configuration"] = (
            ast.literal_eval(config["value"]) if config.get("ok") and config.get("value") else config
        )
        cell = self.ctl(
            "eval",
            code="{k: getattr(rss.get_cell(CELL), k) for k in ('name','center','half','surface_z','ground_z')} if 'CELL' in globals() else None",
        )
        result["work_surface"] = ast.literal_eval(cell["value"]) if cell.get("ok") and cell.get("value") else cell
        result.update(
            self.read_value("__import__('scripts.mimo_chat', fromlist=['observe_objects']).observe_objects(sim)")
        )
        result["arms"] = self.read_value(
            "{name: {'tcp_world': panel.ik.ee_pose(sim.state)[0], "
            "'quat_xyzw': panel.ik.ee_pose(sim.state)[1], 'home': panel.ik.home_pose} "
            "for name, panel in session.arms.items()}"
        )
        return result

    def dispatch(self, name, args):
        if name == "inspect_scene" and not args:
            return self.inspect()
        if name == "read_scene" and not args:
            return {"source": SCENE.read_text()}
        if name == "edit_scene":
            self.phase = "Saving and rebuilding the scene"
            status = self.ctl("status")
            if status["playing"]:
                raise ValueError("Wait for the running routine to finish before editing.")
            old = SCENE.read_text()
            new = apply_replacements(old, args["replacements"])
            if new == old:
                return {"ok": True, "changed": False}
            self.revision += 1
            (self.out / f"scene_{self.revision:03}_before.py").write_text(old)
            (self.out / f"scene_{self.revision:03}_after.py").write_text(new)
            temp = SCENE.with_suffix(".py.pending")
            temp.write_text(new)
            temp.replace(SCENE)
            began = time.monotonic()
            while time.monotonic() - began < 100:
                time.sleep(0.2)
                now = self.ctl("status")
                if now["generation"] > status["generation"]:
                    # The live browser already shows the rebuilt scene. A
                    # separate still adds no model input and can block tools
                    # on native renderer startup or an inactive browser tab.
                    return {
                        "ok": not bool(now["last_error"]),
                        "rebuilt": True,
                        "reset": True,
                        "source_sha256": hashlib.sha256(new.encode()).hexdigest(),
                        **self.inspect(),
                    }
                if now["last_error"] and now["last_error"] != status["last_error"]:
                    return {"ok": False, "error": now["last_error"]}
            raise TimeoutError("Scene rebuild did not complete.")
        if name == "run_routine":
            self.phase = "Robot is running: " + args["name"]
            before = self.inspect()
            reply = self.ctl("play", name=args["name"])
            if not reply.get("ok"):
                return reply
            began = time.monotonic()
            while time.monotonic() - began < 600:
                time.sleep(0.25)
                status = self.ctl("status")
                if not status["playing"]:
                    result = {
                        "ok": not bool(status["last_error"]),
                        "motion_completed": not bool(status["last_error"]),
                        "seconds": round(time.monotonic() - began, 2),
                        "before": before,
                        "after": self.inspect(),
                    }
                    try:
                        measured = self.read_value(
                            f"(lambda fn: fn(sim) if callable(fn) else None)(globals().get('outcomes', {{}}).get({args['name']!r}, globals().get('outcome')))"
                        )
                        if measured is not None:
                            result["metrics"] = measured
                    except Exception as exc:
                        # The motion already happened. Keep its observed state
                        # even if the task's outcome function could not report.
                        result.update(ok=False, measurement_error=str(exc))
                    return result
            # A timeout must not silently leave a routine running.
            self.ctl("reset")
            raise TimeoutError("Routine exceeded 600 seconds and was stopped by reset.")
        raise ValueError(f"Unknown tool or invalid arguments: {name}")

    def run(self):
        try:
            for _ in range(10):
                message = self.completion()
                self.messages.append(message)
                if not message.get("tool_calls"):
                    break
                for call in message["tool_calls"]:
                    name = call["function"]["name"]
                    args = json.loads(call["function"]["arguments"])
                    self.emit("tool_start", call_id=call["id"], name=name, arguments=args)
                    try:
                        result = self.dispatch(name, args)
                    except Exception as exc:
                        result = {"ok": False, "error": str(exc).replace(self.token, "[REDACTED]")}
                    self.emit("tool_end", call_id=call["id"], name=name, result=result)
                    self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
            else:
                raise RuntimeError("Stopped at the 10-request limit for this turn.")
            self.phase = "Ready"
        except Exception as exc:
            self.phase = "Stopped"
            self.emit("error", text=str(exc).replace(self.token, "[REDACTED]"))
        finally:
            with self.lock:
                self.busy = False
            self.emit("done", budget=self.budget.summary())


def handler(agent, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, content, kind="application/json"):
            data = content if isinstance(content, bytes) else json.dumps(content).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                html = (ROOT / "scripts/recording/mimo_chat.html").read_text().replace("__STUDIO_URL__", agent.live_url)
                self.send(200, html.encode(), "text/html; charset=utf-8")
            elif self.path.startswith("/state"):
                from urllib.parse import parse_qs, urlparse

                after = int(parse_qs(urlparse(self.path).query).get("after", ["-1"])[0])
                self.send(200, agent.state(after))
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/chat":
                self.send(404, {"error": "Not found"})
                return
            if self.headers.get("Origin") not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                self.send(403, {"error": "Use the local chat page."})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0 or length > 20000:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                agent.submit(body["message"])
                self.send(202, {"ok": True})
            except (ValueError, KeyError) as exc:
                self.send(400, {"error": str(exc)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--budget", type=float, default=0.25)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument(
        "--resume", action="store_true", help="restore a completed session without resetting its cost ledger"
    )
    parser.add_argument(
        "--allow-budget-increase",
        action="store_true",
        help="explicitly authorize raising a resumed ledger's total limit, preserving spent costs",
    )
    args = parser.parse_args()
    if args.budget <= 0:
        parser.error("--budget must be positive")
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be positive")
    if args.allow_budget_increase and not args.resume:
        parser.error("--allow-budget-increase requires --resume")
    args.out.mkdir(parents=True, exist_ok=True)
    # One writer per ledger, even if somebody tries to resume on another port.
    lock = (args.out / "session.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    agent = Agent(
        args.out.resolve(),
        args.budget,
        max_output_tokens=args.max_output_tokens,
        resume=args.resume,
        allow_budget_increase=args.allow_budget_increase,
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(agent, args.port))
    print(f"Chat: http://127.0.0.1:{args.port}  Model: {MODEL}  Budget: ${args.budget:.2f}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
