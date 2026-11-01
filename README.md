# CholeScope

Try two surgical action sequences from the same starting image. CholeScope
adds mouse and keyboard input, recorded A/B comparisons, and action exports to
[NVIDIA Cosmos3-H-Dreams](https://huggingface.co/nvidia/Cosmos3-H-Dreams).
The scene is robotic cholecystectomy, with a left grasper and right hook.

**Built on NVIDIA Cosmos.** CholeScope uses pretrained weights for research
demonstrations. It is not a clinically validated simulator.

https://github.com/user-attachments/assets/d4c4cf89-9224-44ae-8c5b-87600c66102c

**Live demo — 60 seconds, 2× speed**
· [Download MP4](assets/cholescope-live-demo.mp4)
· [Animated preview](assets/cholescope-live-preview.gif)
· [Poster](assets/cholescope-live-demo.jpg)

An uncut two-minute session: record A, reset, try an opposing input as B,
compare the results, and export the actions. Playback is 2×; inference and
waiting time are retained. Model warmup precedes the take. Captions and a
visible cursor were added for presentation.

*Scene: CMR / Open-H-Embodiment, CC BY 4.0.
[Attribution and video details](THIRD_PARTY.md#live-interface-demo).*

[Preview](#try-the-interface) · [Compare A/B](#compare-two-continuations) ·
[Controls](#mouse-and-keyboard-controls) · [GPU setup](#generate-on-a-gpu) ·
[RunPod](#start-and-stop-an-existing-runpod-pod) · [Performance](#performance)

## Try the interface

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra test
uv run cholescope --preview
```

Open [localhost:8790](http://127.0.0.1:8790). Preview needs no GPU;
inference and recording are disabled. Add `--scene-dir /path/to/scenes` to
show your own image, or `--upstream-root /path/to/cosmos3-h-dreams` to use
NVIDIA's attributed example.

To generate video, follow [GPU setup](#generate-on-a-gpu).

## Compare two continuations

With a GPU connected:

1. Choose a scene and seed, then **Start session**.
2. Select **Record A**, move an instrument, then **Finish A**.
3. Select **Reset for B**, then **Record B**. Try a different input and **Finish B**.
4. Select **Compare A and B** for synchronized playback.
5. Download each branch's **Video** and **Actions**. Input-event export is under **Sources & limits**.

Each recording resets the image, seed, model history, and controller pose.
B starts from the original scene. Recordings run sequentially on one model
session and are capped near 30 seconds, rounded to a full chunk. Matching
seeds helps comparison but does not guarantee identical results across systems.

Exports include model actions, frame count, seed, image checksum, model
settings, expected source revision, and a generated-content flag. Keep the
source image, model manifest, and runtime settings alongside them.

## Mouse and keyboard controls

Select the **Left grasper** or **Right hook**, then drag inside the live view.
Drag horizontally or vertically to request movement; scroll **while dragging**
for depth. Hold **Shift** for finer input. The ring and tether show your
gesture, not a tracked tool tip. Commands do not guarantee matching movement
in the generated image.

For keyboard input, focus the page outside a form control. Directions follow
the model's axes, which may differ from screen directions.

| Key | Action |
| --- | --- |
| `1` / `2` | Select left / right instrument |
| `A` / `D` | Translate along the first axis |
| `Q` / `E` | Translate along the second axis |
| `S` / `W` | Translate along the third axis |
| Up / Down | Pitch |
| Left / Right | Yaw |
| `Z` / `X` | Roll |
| `F` / `V` | Close / open the jaw |
| Hold `Shift` | Fine motion at one-quarter speed |
| Space | Pause generation |

Releasing a key or drag holds the requested pose. Reset restores the initial
pose. Losing focus releases held keys and discards unsent drag motion;
accepted pose changes remain. Energy and camera controls stay zero. These
inputs control a model, not physical robot hardware.

<details>
<summary>Input timing and diagnostics</summary>

One video-width of dragging requests 0.5 model units on either screen axis,
or 0.125 with Shift. Distance sets the command, regardless of drag speed.
Large gestures are split into bounded packets; requested translation is
clamped to ±1 model unit per axis.

Keyboard motion uses elapsed server time. A separate 10 Hz loop samples poses
while inference runs, and stale held keys expire after 350 ms. The pending
buffer holds one chunk. Under load, newer poses replace older samples, so
intermediate motion may be skipped. Pause clears the buffer and waits for
in-flight inference before a reset or resume.

Diagnostics report sampling, inference, and publication timestamps plus
`skipped_action_samples`. These measure server behavior, not browser latency
or visual control accuracy.

</details>

## Coordinate student lab

Open **Coordinate** (`/student`) on the running server. Choose a target, then
select **Watch untrained → Train student → Watch trained**. Both clips use
the same scene and seed. **Reset learning** restarts the experiment.

The lab trains a small controller to issue geometric reaching commands;
**Cosmos stays frozen**. The map and score measure commanded coordinates,
not the visible tool tip. Use the same browser tab when switching modes:
Manual and Coordinate share one operator connection.

<details>
<summary>What the student learns</summary>

The student is a 1,218-parameter `2 → 32 → 32 → 2` tanh network. It takes the
target's offset from the commanded XY position and outputs a bounded left-arm
translation. Depth, wrist, jaws, the other arm, and energy stay fixed.

Training runs 800 Adam updates on CPU using synthetic examples from an
analytic reaching controller. This is supervised imitation of that controller;
Cosmos provides neither training observations nor rewards. Generated motion
may drift or distort.

The lab reports batch losses and 25 held-out target rollouts, with stationary
and analytic baselines. These test the coordinate task, not generalization to
anatomy. Downloads include before/after weights, update counts, and hashes;
action exports also identify the checkpoint, goal, image, and task scope.
Trials last 64 action steps (6.4 seconds); interrupted trials are marked incomplete.

</details>

## Generate on a GPU

Requires Linux, Python 3.12, `git`, uv 0.11.8+, and an NVIDIA driver compatible
with the pinned CUDA 13 environment. Upstream reports **more than 32 GB VRAM**
for its default configuration; this project has been tested on a 96 GB
RTX PRO 6000. See [upstream requirements](https://github.com/isaac-for-healthcare/cosmos3-h-dreams#system-requirements).

Keep dependencies, weights, caches, and outputs on a mounted asset disk.
For a rented pod, [arm the budget watchdog](#bound-a-runpod-session) before setup.

<details>
<summary>Install dependencies, download weights, and launch</summary>

Set paths for your asset disk:

```bash
export CHOLESCOPE_ASSET_DIR=/mnt/cholescope-assets
export CHOLESCOPE_UPSTREAM=/mnt/cholescope-assets/source/cosmos3-h-dreams

python3 scripts/bootstrap_gpu.py \
  --asset-dir "$CHOLESCOPE_ASSET_DIR" \
  --upstream-root "$CHOLESCOPE_UPSTREAM"

"$CHOLESCOPE_UPSTREAM/.venv/bin/python" scripts/download_assets.py \
  --asset-dir "$CHOLESCOPE_ASSET_DIR"
```

Bootstrap installs the locked upstream workspace at commit
`fb18792d8af680a713801dea7688919e46329810`, CholeScope, and the portable
`imageio-ffmpeg==0.6.0` encoder. Without `--upstream-root`, source goes in
the ignored `.vendor/cosmos3-h-dreams` directory.

The downloader fetches the two-step BF16 checkpoint, Wan VAE, optional TAEHV
decoder, Qwen tokenizer, and license notices. Revisions and paths go in
`asset-manifest.json`. Use `--model-revision <commit>` to pin weights,
`--dependencies-only` to reuse a checkpoint, or `--dry-run` to inspect downloads.
Keep weights outside Git.

Read the checkpoint path from the manifest and start the server:

```bash
export CHOLESCOPE_CHECKPOINT="$(python3 - <<'PY'
import json, os
from pathlib import Path
manifest = json.loads((Path(os.environ['CHOLESCOPE_ASSET_DIR']) / 'asset-manifest.json').read_text())
files = manifest['repositories']['nvidia/Cosmos3-H-Dreams']['files']
print(files['checkpoints/exp15/c3-h-d_2-step_model_bf16.safetensors']['path'])
PY
)"

python3 scripts/launch_gpu.py \
  --asset-dir "$CHOLESCOPE_ASSET_DIR" \
  --upstream-root "$CHOLESCOPE_UPSTREAM" \
  --checkpoint "$CHOLESCOPE_CHECKPOINT" \
  --offline
```

On RunPod, also supply `--lease /path/to/private/lease.json`, `RUNPOD_POD_ID`,
and `RUNPOD_API_KEY`. The launcher starts a second watchdog and requests the
lease's shutdown action on exit. Download recordings before exiting.

`--offline` prevents Hugging Face refreshes after download. Caches stay on the
asset disk. The first scene requires GPU compilation and takes longer to prepare.

</details>

Connect through SSH, then open [localhost:8790](http://127.0.0.1:8790):

```bash
ssh -N -L 8790:127.0.0.1:8790 ssh-user@gpu-host
```

For an HTTPS reverse proxy, `--host 0.0.0.0` requires `CHOLESCOPE_ACCESS_TOKEN`
in the environment. Use any HTTP Basic username and the token as the password.
Keep credentials out of source files and command arguments.

## Start and stop an existing RunPod pod

The optional local launcher keeps the interface open while the GPU is off.
**Start pod** resumes the configured pod, opens an SSH tunnel, and loads the
cached model. **Stop pod** finishes the recording and stops the pod. Download
recordings first; closing the browser does not stop the GPU.

Prepare the GPU environment above on a persistent or network volume.
RunPod's temporary container disk is cleared on restart, even if named
`/workspace`. Resuming also requires GPU capacity on the pod's host.

<details>
<summary>Configure and run the local launcher</summary>

Keep keys, configuration, and budget state outside the checkout. Key files
must have permissions `0600`. Save this example as `/path/to/private/runpod.json`
and replace its identity, paths, and spending baseline:

```json
{
  "pod_id": "your-existing-pod-id",
  "pod_name": "your-exact-pod-name",
  "created_at": "2026-01-01T00:00:00Z",
  "key_file": "/path/to/private/runpod-api-key",
  "ssh_key": "/path/to/private/ssh-key",
  "known_hosts": "/path/to/private/known_hosts",
  "state_dir": "/path/to/private/demo-state",
  "initial_spend_usd": 0,
  "initial_spend_epoch": 1767225600,
  "remote_repo": "/workspace/cholescope",
  "remote_assets": "/workspace/cholescope-assets",
  "remote_private": "/workspace/cholescope-private",
  "session_minutes": 60,
  "max_rate": 3,
  "storage_reserve": 5,
  "paused_storage_rate": 0.05,
  "remote_guard": false,
  "allow_replacement": false
}
```

`initial_spend_epoch` is the UTC Unix timestamp through which spending has
been accounted. Adjust the stopped-storage rate for your disk. These are
operator estimates, not RunPod billing data.

```bash
uv run python scripts/local_demo.py \
  --config /path/to/private/runpod.json \
  --upstream-root /path/to/cosmos3-h-dreams
```

Open [localhost:8792](http://127.0.0.1:8792). Credentials stay out of the
browser; controls bind to loopback, reject cross-origin requests, and use SSH
for model traffic.

</details>

The launcher arms a local watchdog before starting paid compute. Sessions
are capped at 60 minutes, with shutdown requested two minutes early.
**Keep the controller awake and connected until the pod stops.** macOS idle
sleep is prevented, but closing the lid or losing connectivity can interrupt
the guard. `"remote_guard": true` adds a GPU-host watchdog and requires
explicit approval to store the API key there.

The persistent ledger reserves the full session at the configured maximum
rate, even when stopped early. This is **an estimate, not an invoice**. It
includes estimated storage, enforces the $80 project allowance with at least
$5 reserved for storage, and must not be reset between sessions. A definitive
rejection before compute starts releases the reservation; an ambiguous result
keeps it reserved. Reconnection does not extend the deadline. Stopped disks
may still incur charges, and API outages can delay shutdown.

Only one tab controls the model. **Use this tab** transfers control without
restarting the pod or extending the session.

<details>
<summary>Optional fallback when the pod's host has no capacity</summary>

Add these fields to the private configuration. Use your existing 150 GB
standard network volume and the public key corresponding to `ssh_key`:

```json
{
  "allow_replacement": true,
  "replacement": {
    "gpu_id": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "data_center_id": "EU-RO-1",
    "volume_id": "your-network-volume-id",
    "image": "runpod/pytorch:1.4.1-cu1320-torch280-ubuntu2404",
    "public_key": "ssh-ed25519 REPLACE_WITH_YOUR_PUBLIC_KEY",
    "max_hourly_rate": 2.49
  }
}
```

After a definitive capacity rejection, the launcher may create **one replacement
per Start click**. It verifies the old pod is stopped and checks the configured
region, GPU, price, and volume. Capacity is not guaranteed.

The replacement uses the same volume, one 96 GB GPU, a 30 GB temporary disk,
SSH port 22, and the remaining session time and budget reservation. Old pods
stay stopped; no pods or volumes are deleted. The browser cannot change this recipe.

A local allocation guard starts before creation. A lost response is reconciled
by the unique attempt name, not retried. **Stop pod** cancels unresolved
allocation, including a pod that appears later. Keep the controller awake and
connected until the remote watchdog can start.

</details>

## Performance

On 24 October 2026, an RTX PRO 6000 Blackwell (96 GB) completed **50 rollouts /
3,200 frames** with the frozen two-step BF16 checkpoint, full Wan decoder,
and 832 × 480 output.

| Frames per chunk | Median chunk-ready latency | 95th percentile | Generation-only throughput |
| --- | --- | --- | --- |
| 8 | 1.177 s | 1.200 s | 16.75 FPS |
| 4 | 0.571 s | 0.585 s | 14.81 FPS |

Four-frame chunks reduced median chunk-ready latency by **51.5%**. Latency
runs from the first action sample to the decoded chunk; it excludes network
transfer and playback. Throughput excludes input sampling and is not the
delivered frame rate. Lower latency does not establish accurate visual control.

The standalone default is **eight-frame chunks with the full Wan decoder**.
For experimental lower latency, use `--chunk-frames 4` on the server or GPU
launcher, or `"chunk_frames": 4` in the private pod configuration before a
new session. Set it to `8` to restore the baseline. Output changes with this
setting: tool shape and texture can drift, and small Y commands often produced
little visible motion.

<details>
<summary>Benchmark method and earlier decoder results</summary>

Five images came from separate CMR recordings: three with the intended two
tools and two with extra instruments. Each configuration ran 25 rollouts with
identical 64-step action sequences and seed `42`. The images are not proven
held out from Cosmos pretraining. Measurements include history-cache filling.
Both configurations remove the former 100 ms trailing wait per chunk.

Wan preserves more detail than the faster, softer TAEHV decoder. Use
`--decoder taehv` on the direct server to select TAEHV. The earlier animation
below used TAEHV and measured 34.7 FPS with about 1.05 s chunk-ready latency.
A separate single-scene check (two seeds, 192 frames each, steady-state chunks)
measured 40 FPS generation-only for eight-frame chunks and 32 FPS for four. Latencies were
1.00 s and 0.53 s under the old cadence; subtracting the removed wait gives
estimated 0.90 s and 0.43 s, not new measurements. These runs are not directly
comparable to the five-scene benchmark.

The checkpoint was trained at 832 × 480. In the earlier example-scene check,
960 × 544 looked no sharper and 1280 × 720 turned hazy.

</details>

<details>
<summary>Earlier generated A/B example</summary>

![AI-generated surgical A/B continuations from the same starting image and seed](assets/cholescope-comparison.gif)

*Same CMR starting image and seed `42`, with opposing fine left-grasper
commands along the first model axis. Source: CMR / Open-H-Embodiment,
CC BY 4.0; generated continuations, resizing, and annotation.
[Media attribution](THIRD_PARTY.md#published-demo-animations).*

</details>

<details>
<summary>Reproduce the benchmark</summary>

### Repeat the responsiveness comparison

`scripts/benchmark_responsiveness.py` uses cached weights and never provisions
a GPU. Keep the rental watchdog running. Prepare a manifest with a `scenes`
array; each entry needs `id`, image `path` relative to the manifest,
`source_url`, `source_recording_id`, image `sha256`, and `license`.
Images are hash-checked before loading.

Run each chunk size in a separate process and use new output directories:

```bash
for chunk in 8 4; do
  "$CHOLESCOPE_UPSTREAM/.venv/bin/python" scripts/benchmark_responsiveness.py \
    --manifest /path/to/scene-manifest.json \
    --output-dir "$CHOLESCOPE_ASSET_DIR/benchmarks/chunk-$chunk" \
    --asset-dir "$CHOLESCOPE_ASSET_DIR" \
    --upstream-root "$CHOLESCOPE_UPSTREAM" \
    --checkpoint "$CHOLESCOPE_CHECKPOINT" \
    --decoders wan --chunk-frames "$chunk" --seeds 42 --frames 64 \
    --max-wall-seconds 1050
done
```

`--dry-run` validates the manifest and case count. Default cases use stationary
and opposing X/Y commands with matched actions. Outputs include hashes,
timings, videos, contact sheets, and incomplete or failed cases. The wall-time
reserve prevents new work near the limit but cannot interrupt a stuck CUDA call.
Inspect matched videos and stationary drift before choosing a faster setting.

</details>

## Bound a RunPod session

The budget utility guards an existing pod; it never creates one. Use the
actual total hourly quote, including all GPUs and prior project spending.
The following rate is an example, not a live offer.

<details>
<summary>Create a lease and start the independent watchdog</summary>

```bash
python3 scripts/runpod_budget.py quote \
  --hourly-rate 2.49 --hours 12 --spent 0
```

The guard permits at most 12 hours, projected compute spending below $75,
and at least $5 reserved for storage within an $80 total. It requests shutdown
two minutes early. The storage reserve is an estimate, not a billing cap.

After provisioning, copy the exact pod ID, name, and `createdAt` into a lease
on an independent, connected controller. Store it outside the repository:

**`terminate` deletes the pod and its local disks.** Download recordings first.
Use `--action stop` to preserve an attached persistent Pod volume; temporary
container storage is still erased. Separate network volumes survive both
actions, and retained storage can still incur charges.

```bash
python3 scripts/runpod_budget.py lease \
  --pod-id "YOUR_POD_ID" --pod-name "YOUR_POD_NAME" \
  --created-at "YYYY-MM-DDTHH:MM:SSZ" \
  --hourly-rate 2.49 --hours 12 --spent 0 \
  --action terminate --output /path/to/private/lease.json

python3 scripts/runpod_budget.py watch \
  --lease /path/to/private/lease.json --pod-id "YOUR_POD_ID"
```

Supply `RUNPOD_API_KEY` through the environment or a secret manager. Keep the
watchdog running through setup and use. Copy the lease to the pod and pass it
to the GPU launcher, which checks identity and starts another watchdog.
The application process does not receive the API key.

</details>

The watchdog is not a provider-enforced spending cap. Sleep, lost connectivity,
or API failures can delay shutdown and extend billing. It retries and verifies
cloud state; check RunPod after shutdown. The integration uses the
[RunPod REST v2 API](https://docs.runpod.io/api-reference-v2/pods/get-a-pod).

## Validation and limits

```bash
uv run pytest
node --test src/cholescope/static/*.test.mjs
python3 -m unittest discover -s scripts -p 'test_*.py'
python3 scripts/check_repo_hygiene.py
```

CPU tests cover controls, sessions, model-adapter contracts, and budget failure
handling. To check a GPU setup, generate both branches, export playable MP4s,
and inspect tool deformation, anatomy drift, and responses to opposing inputs.

The model can invent anatomy and tissue responses. It provides no validated
forces, collisions, cutting mechanics, or patient-specific physical state.
**Do not use it for treatment decisions, surgical guidance, or autonomous
hardware control.** Its final clinical training data are proprietary; public
Open-H input scenes do not reproduce that training.

CholeScope source is [Apache-2.0](LICENSE). Weights, images, and dependencies
retain their own terms. See [third-party attribution](THIRD_PARTY.md).
