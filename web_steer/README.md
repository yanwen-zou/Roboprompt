# RoboPrompt Web Steer

Browser-based alternative to the OpenCV interactive prompt window. The Flexiv robot client publishes base/wrist observations here, converts submitted browser annotations back to NumPy prompt inputs, and includes them in its next policy-server inference request. Local and browser prompt entry are both supported.

## Run

Complete [Quick Start](../README.md#quick-start) first. From the repository root:

```bash
python web_steer/server.py --host 0.0.0.0 --port 8765
```

Open `http://<server-ip>:8765`.

Run the real robot client with browser prompts:

```bash
python openpi/examples/flexiv_real/main.py \
  --args.steer evo \
  --args.prompt-ui web \
  --args.web-steer-url http://127.0.0.1:8765
```

Use `--args.prompt-ui local` (the default) to retain the existing `P` hotkey and OpenCV prompt window.

## GitHub Pages + local Python server

The Pages site hosts only HTML/CSS/JS. Images and prompts flow directly between
the browser and your local Python server, not through GitHub. The existing UI
accepts visual points/trajectories and global movement, not free-form text.

1. Commit/push the changes (including `.github/workflows/web-steer-pages.yml`) to
   the repository's default branch. In GitHub **Settings → Pages → Build and
   deployment**, choose **GitHub Actions**. In **Actions**, manually run
   **Deploy Web Steer to GitHub Pages**. The workflow publishes only the three
   static frontend files. It does not start Python or run the robot. Running this
   workflow replaces this repository's current Pages site if it already has one.
2. On the robot computer, start the bridge, replacing `YOUR_USER` with the Pages
   owner (or use the exact origin of your custom domain):

   ```bash
   .venv/bin/python web_steer/server.py \
     --host 127.0.0.1 --port 8765 \
     --allowed-origin https://YOUR_USER.github.io
   ```

   The allowed origin must **not** include `/Roboprompt/`. Repeat
   `--allowed-origin` to allow additional sites. Same-origin local UI access and
   Python clients work without this option. Cross-origin requests from other
   websites are rejected before modifying API state. This origin check is not
   authentication for non-browser clients; the default listener is loopback.
3. Start your usual steering policy server. In the same robot environment you
   normally use, start the robot client with web prompt entry:

   ```bash
   PROMPT_UI=web WEB_STEER_URL=http://127.0.0.1:8765 \
     bash scripts/realworld/eval/evo1/client/run_client_openpi.sh
   ```

   Retain your normal `POLICY_HOST`, `POLICY_PORT` and other robot settings.
   Direct Python invocation can instead use `--args.steer evo
   --args.prompt-ui web --args.web-steer-url http://127.0.0.1:8765`.
4. Open the workflow's Pages URL **on the robot computer**. The page connects
   automatically (default `http://127.0.0.1:8765` on Pages); the address field
   is not shown. To override the address, open
   `https://YOUR_USER.github.io/Roboprompt/?server=http%3A%2F%2F127.0.0.1%3A8765`.
   Allow local-network/loopback access if the browser asks. Once the camera is
   visible, submit an annotation and check that **Inference Status** becomes
   `#N Used for inference`; that acknowledgement means policy inference returned, not that
   the robot has completed the motion.

Browser support for HTTPS-to-local-HTTP access varies. If connection is blocked,
check the site's local network permission, server origin configuration and the
browser console. See [MDN local network access](https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/Local_network_access).
You can always use `http://127.0.0.1:8765` directly for the same-origin UI.

If the browser runs on another computer, `127.0.0.1` points to that computer,
not the robot. One option is an SSH forward on the browser computer:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@ROBOT_HOST
```

Then use `?server=http://127.0.0.1:8765` on the Pages URL. Alternatively,
bind the bridge to a reachable interface and use its address; HTTPS Pages access
to LAN HTTP depends on browser local-network support, and a trusted HTTPS
reverse proxy may be needed. Merely publishing on Pages does not expose the
local Python server to the internet.

Cross-origin UI state/images refresh every second. The server stores only the
latest prompt in memory; it is not a durable command queue. Start the client
before submitting prompts, as the client skips prompts present at startup.

## Phone on the same Wi-Fi

Restart the bridge on the robot computer with a LAN listener:

```bash
.venv/bin/python web_steer/server.py \
  --host 0.0.0.0 --port 8765 \
  --allowed-origin https://YOUR_USER.github.io
```

Keep the robot client's `WEB_STEER_URL=http://127.0.0.1:8765`. On the phone,
open `http://ROBOT_WIFI_IP:8765` directly. The page automatically connects to
that address. The local server serves the same UI; API calls stay on the same
origin. An explicit `?server=http://ROBOT_WIFI_IP:8765` also sets the address.
`127.0.0.1` on the phone refers to the phone, and `0.0.0.0` is a listener
address, not the address to enter in the browser. Both devices must have a
reachable LAN connection (guest Wi-Fi/client isolation can prevent this).
Use this listener only on a trusted network; the bridge has no authentication.

## Prompt appearance

After a successful submission, the submitted sketch is cleared from the canvas.
Failed submissions keep the sketch for retry, and marks drawn while a submission
is pending are retained. Clearing the canvas does not cancel the submitted robot
prompt. Undo and clear controls are also available below the tools on phones.

Web preview and exported prompt images share one renderer: white-to-orange
RGB(255, 80, 0) trajectory segments without arrowheads, and orange target
disks with white borders, matching `scripts/utils/draw_overlay.py`. Sizes use
the desktop UI's nominal 520px image height (line radius 1, point radius 6,
outer radius 8) and scale with the image on any screen. The desktop UI may
reduce its drawing canvas to fit the monitor; browser antialiasing also differs
from the Python disk rasterizer, so output is not pixel-identical.

Validation (without robot hardware):

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  web_steer/test_server.py scripts/utils/web_steer_client_test.py -q
node --check web_steer/app.js
node --test web_steer/render_test.cjs
```

## Client API

Upload an observation as multipart data:

```python
import json
import requests

with open("base.jpg", "rb") as base, open("wrist.jpg", "rb") as wrist:
    requests.post(
        "http://127.0.0.1:8765/api/observation",
        files={"base_image": ("base.jpg", base, "image/jpeg"), "wrist_image": ("wrist.jpg", wrist, "image/jpeg")},
        data={"metadata": json.dumps({"frame_id": 42, "state": [0.0] * 8})},
        timeout=2,
    ).raise_for_status()
```

Poll for a new prompt (the request can long-poll for up to 30 seconds):

```python
sequence = 0
response = requests.get(
    "http://127.0.0.1:8765/api/prompt",
    params={"after": sequence, "wait": 30},
    timeout=35,
)
if response.status_code == 200:
    prompt = response.json()
    sequence = prompt["sequence"]
```

The returned JSON includes `prompt_images.prompt_0` as a PNG data URL. Decode it to an RGB ndarray before passing the payload to code that expects `numpy.ndarray`; vectors and masks already use the existing names (`prompt_2d_drag`, `prompt_global_motion`, etc.). JSON/base64 observation uploads are also supported with `base_image` and optional `wrist_image` fields.

## Endpoints

- `POST /api/observation` — publish a camera observation.
- `GET /api/state` — current frame metadata and UI state.
- `GET /api/observation/base|wrist` — current image bytes.
- `POST /api/prompt` — browser submits a prompt.
- `GET /api/prompt?after=N&wait=30` — client reads the next prompt.
- `POST /api/prompt/N/ack` — client confirms prompt N was included in policy inference.
- `DELETE /api/prompt` — clear the current prompt.
- `GET /healthz` — health check.
