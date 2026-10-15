# live: a page that rebuilds `scene.py` every time you save it

Point a coding agent at this folder: it edits `scene.py`, the open page
rebuilds on every save, and the agent checks the result with snapshots. The
page is the studio, so you can drag things on it between edits.

What is here:

- `scene.py`: the starter scene and the agent's working file. Bind `routine`
  (or `routines = {name: fn}`) to get a Run button.
- `ctl`: the command line onto the running page; it also runs from the
  repository root as `examples/live/ctl`. It uses the checkout's `.venv`, else
  `$PYTHON` or `python3`, so it also works after
  `pip install "robosimstudio @ git+https://github.com/robosimstudio/robosimstudio"`.
- `run/`: made on first use and not committed. It holds the socket, the
  server's log and every snapshot.

## Start

```bash
cd examples/live
./ctl up                                    # the page; prints its URL, http://localhost:8080 or the next free port (--share adds a public URL)
claude --permission-mode acceptEdits        # Claude Code; edits go through without a prompt
codex -a on-request -s workspace-write      # or Codex; the sandbox may write here and reach the socket
```

The page opens on a shared tabletop with two YAM arms, two cubes, exchange and
destination pads, cloth, cable and a soft block. On a remote GPU
box, pick an idle card with `CUDA_VISIBLE_DEVICES` and use `--share` or an SSH
tunnel (`ssh -L 8080:localhost:8080 <box>`). The page listens on all interfaces
(viser's default), so on a shared network firewall the port or use the tunnel.
`./ctl quit` stops the server.

## A model operating through chat

The local chat page sends messages to MiMo through Hugging Face Inference
Providers. The model reads the scene and live state, edits `scene.py`, and runs
the routines it chooses. The browser displays the model's response, tool results
and estimated spending alongside the live simulation.

From the repository root, with `HF_TOKEN` already in the environment:

```bash
examples/live/ctl up --hold
uv run python scripts/mimo_chat.py --out outputs/my-chat --budget 1
```

Open the printed chat URL (normally `http://127.0.0.1:8092`) and send, for example:

> Inspect the table, choose a short interaction with one object, and carry it
> out using robot contact. Keep everything on the table and measure the result.

The model receives text observations: body and particle positions, bounds,
sampled geometry, solver ownership and routine outcomes. It does not receive
camera images. Its source edits rebuild and reset the scene; the page keeps
the current viewpoint and drawer layout while the replacement arrives.

The current workbench has two YAM arms, two rigid blocks, a central exchange
pad, destination pads, cloth, cable and foam. The arms and rigid blocks share
MuJoCo's native contacts, allowing friction grasps and lifts. Cloth, cable and
foam share a VBD island with Newton's two-way proxy coupling to MuJoCo.
Cloth pinching remains unsupported under that coupling. `CAPABILITIES` and
the live observations describe the available contact behavior to the agent.

The checked-in `handoff` and `reverse_handoff` routines were authored and
repaired by MiMo through this chat. The first passes the red cube from left to
right via the exchange pad; the second passes the teal cube in the opposite
direction. Both measure each lift, exchange placement, final placement and
resting drift from simulated object positions. For a repeatable demonstration,
start from a reset scene and ask for the red handoff, then the teal handoff.
Neither routine needs a scene edit between runs.

The live page starts in photoreal RTX mode with editing overlays hidden.
OVRTX and an RTX-capable GPU are required for this view. It keeps the last
rendered image while rebuilding and preserves the camera and render settings.
A renderer failure reports an error and retains the last frame; it does not
switch to collider or browser rendering. The recorder waits for actual RTX
frames before marking a take ready and rejects a failed renderer.

The default provider is `XiaomiMiMo/MiMo-V2.6-Pro-RL:novita`. Pricing is fetched
at startup. Before each request, the server reserves a conservative allowance
for the full input and up to 32,768 output tokens, including reasoning. It then
records the provider's actual token usage. Estimated costs price cached input
at the full rate, so the HF invoice can be lower. Unknown usage keeps its
reservation, and a truncated response never executes partial tool calls.
The output allowance is configurable with `--max-output-tokens`; it is a ceiling,
not a target. Requests have a 30-minute overall deadline and a 60-second network
idle timeout. There is no automatic retry after truncation or a timeout.

`--budget` is the total limit for the saved session. Restarting preserves both
spending and the conversation; it never replays robot actions:

```bash
uv run python scripts/mimo_chat.py --out outputs/my-chat --budget 1 --resume
```

Raising an existing limit also requires `--allow-budget-increase`. Each change
is recorded in `usage.json`. The output directory retains the request/response
records, tool events, source revisions and usage ledger; credentials and private
model reasoning are excluded.

To record the actual chat and simulation in a separate visible browser:

```bash
uv run --group media python scripts/record_mimo_chat.py --out outputs/my-capture
# Interact through the chat, then finish the capture:
touch outputs/my-capture/stop-recording
```

The recorder submits no messages and invokes no robot tools. It saves the raw
capture, screenshots, timing metadata and `experience.mp4` at real elapsed
speed. Only browser startup is trimmed. Each capture needs a fresh directory.

Both agents read the procedure from [`AGENTS.md`](../../AGENTS.md), so the
prompt only has to say what to build. Start each prompt below with this
line:

```
A live page is running: it rebuilds scene.py on every save. Use ./ctl to inspect it (see ./ctl --help).
```

## The prompt, with a clip

Drop the clip beside the checkout (not in it) and say:

```
../../../<clip>.mov is a real clip of a person doing one thing at a table.
Rebuild it in scene.py with this library's core API: the table, two YAM
arms where the person sat, what is on the table, a camera where the phone
was, what stands behind the table, and then the arms doing what the person
did, bound as `routine` in scene.py so I get a Run button on the page. Look
at the pairs, the look sheets and the strip with your own eyes, not only the
numbers: the things in their detail, the background, what happened. I am
watching the page. When you are done, report the readings and what you saw:
the first frame against the clip, the gate, and the take against the clip's
timeline. Say in one line what you are about to do before each edit.
```

## The prompt, with your own photo

Make a `ref/` folder here, drop your photo into `ref/`, and say:

```
ref/<your photo> is a real photo of a robot bench. Rebuild that scene in
scene.py with this library's core API, one thing per save so I can watch each
one appear on the page: the bench, the robot, what is on the bench, and a
camera where the photo was taken from. Measure sizes and placements from the
photo. After each save, snapshot through your camera beside the photo
(./ctl snapshot --beside ref/<your photo>), look at the pair, and adjust until
they line up. When the scene matches, make the robot do something to an object
and step it on the page so I can watch. Say in one line what you are about to
do before each edit.
```

## The prompt, with no photo

```
Build this in scene.py with this library's core API, one thing per save so I
can watch each one appear on the page: two YAM arms, a towel and a box on a
bench. After each save, take a snapshot and look at it before the next
edit. When the scene is built, make an arm push the box and step it on the
page so I can watch. Say in one line what you are about to do before each
edit.
```

## Where things go

Everything the page writes lands in `run/` (snapshots, comparisons, takes,
`log.jsonl`, `serve.log`); clip stills and `timeline.json` go in `ref/`. The
full list, every `ctl` command, the contract for `scene.py` and the frames and
units are in [`REFERENCE.md`](REFERENCE.md#ctl).

The result is `scene.py` itself. Copy it aside to keep it;
`git checkout examples/live/scene.py` puts the starter back.
