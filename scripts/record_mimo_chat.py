# SPDX-License-Identifier: Apache-2.0
"""Record the real chat UI; this recorder never submits prompts or drives the sim.

The operator can attach to the browser's CDP port and type into the chat. Touch
OUT/stop-recording to finish. The only trim removes page startup; no retiming.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright
from recording.page import CURSOR_JS, GPU_ARGS, HELPER_JS

from robosimstudio_real2sim_agent.wire import request

SOCKET = Path(__file__).resolve().parents[1] / "examples/live/run/ctl.sock"


def photoreal_status():
    try:
        status = request(SOCKET, {"cmd": "status"}, timeout=2)
    except TimeoutError:
        # Rebuilds hold the sim thread while the browser retains its last RTX
        # frame. The readiness deadline below still bounds this wait.
        return {"ready": False, "building": True}
    if not status.get("ok") and "busy" in status.get("error", ""):
        return {"ready": False, "building": True}
    view = status.get("render", {})
    if view.get("mode") != "rtx" or view.get("backend") != "rtx" or view.get("error"):
        raise RuntimeError(f"Recording requires a working photoreal viewport: {view}")
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8092")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cdp-port", type=int, default=9333)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "raw").exists() or (out / "stop-recording").exists():
        raise FileExistsError("Choose a fresh capture directory")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=False, args=[*GPU_ARGS, f"--remote-debugging-port={args.cdp_port}", "--window-size=1920,1080"]
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1,
            record_video_dir=str(out / "raw"),
            record_video_size={"width": 1920, "height": 1080},
        )
        page = context.new_page()
        created = time.monotonic()
        page.goto(args.url)
        page.wait_for_function("window.agentState !== undefined")
        frame = next(f for f in page.frames if f != page.main_frame)
        frame.wait_for_selector("canvas", timeout=60000)
        frame.wait_for_function("!!window.__viserMutable?.camera", timeout=60000)
        frame.evaluate(HELPER_JS)
        renderer = frame.evaluate("() => window.__rss.renderer()")
        if "NVIDIA" not in renderer:
            raise RuntimeError(f"Expected GPU rendering: {renderer}")
        page.evaluate(CURSOR_JS)
        frame.evaluate(CURSOR_JS)
        # Layout only. The live page preserves the drawer across rebuilds;
        # no periodic clicks or camera-gizmo toggles during a recording.
        collapse = frame.get_by_role("button", name="Collapse panel area", exact=True)
        if collapse.is_visible():
            collapse.click()
        deadline = time.monotonic() + 180
        while not photoreal_status().get("ready"):
            if time.monotonic() > deadline:
                raise TimeoutError("Photoreal viewport did not become ready; no demo was recorded.")
            page.wait_for_timeout(500)
        page.wait_for_timeout(2000)
        view = photoreal_status()
        start = time.monotonic()
        page.screenshot(path=str(out / "start.png"))
        (out / "ready.json").write_text(
            json.dumps(
                {
                    "url": args.url,
                    "cdp_port": args.cdp_port,
                    "renderer": renderer,
                    "photoreal": view,
                    "created_monotonic": created,
                    "ready_monotonic": start,
                }
            )
        )
        print(f"Recording; interact through {args.url}. Touch {out / 'stop-recording'} to finish.", flush=True)
        try:
            last_ready = time.monotonic()
            while not (out / "stop-recording").exists():
                page.wait_for_timeout(500)
                if photoreal_status().get("ready"):
                    last_ready = time.monotonic()
                elif time.monotonic() - last_ready > 180:
                    raise TimeoutError("Photoreal viewport stopped delivering frames.")
        finally:
            ended = time.monotonic()
            marker = out / "operator-start.json"
            if marker.exists():
                start = max(start, min(ended, json.loads(marker.read_text())["monotonic"] - 2))
            duration = ended - start
            page.screenshot(path=str(out / "end.png"))
            raw = page.video
            context.close()
            raw_path = raw.path()
            browser.close()
            meta = {
                "raw": str(raw_path),
                "trim_start": start - created,
                "duration": duration,
                "speed": 1,
                "url": args.url,
                "renderer": renderer,
                "created_monotonic": created,
                "started_monotonic": start,
                "ended_monotonic": ended,
            }
            (out / "capture.json").write_text(json.dumps(meta, indent=2))
    with (out / "encode.log").open("w") as log:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(raw_path),
                "-ss",
                str(meta["trim_start"]),
                "-t",
                str(duration),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(out / "experience.mp4"),
            ],
            check=True,
            stdout=log,
            stderr=log,
        )
    print(out / "experience.mp4", flush=True)


if __name__ == "__main__":
    main()
