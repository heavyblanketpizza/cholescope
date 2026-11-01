# Third-party materials

CholeScope's source is licensed under [Apache-2.0](LICENSE). This license does
not relicense downloaded model weights, datasets, or dependencies. Weights,
full datasets, and runtime dependencies are obtained separately. Attribution
for the published demo animations appears below.

**Built on NVIDIA Cosmos.** The NVIDIA Open Model License requires this
attribution for products or services using a Cosmos model.

| Material | Source and terms |
| --- | --- |
| Cosmos3-H-Dreams and its included FlashDreams runtime | [NVIDIA source](https://github.com/isaac-for-healthcare/cosmos3-h-dreams), Apache-2.0; copyright NVIDIA Corporation & Affiliates. Bootstrap retains upstream license and notice files. Pinned commit: `fb18792d8af680a713801dea7688919e46329810`. |
| Cosmos3-H-Dreams surgical checkpoint | [NVIDIA model card](https://huggingface.co/nvidia/Cosmos3-H-Dreams), governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). The downloader selects the two-step BF16 checkpoint. |
| Wan2.2 VAE | [Wan-AI model repository](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B); retain its model card and license. Only the VAE is fetched. |
| TAEHV tiny decoder | [madebyollin/taehv](https://github.com/madebyollin/taehv), with `taew2_2.pth` obtained from the [lightx2v mirror](https://huggingface.co/lightx2v/Autoencoders). Retain upstream code and weight notices. |
| Qwen3-VL tokenizer | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), Apache-2.0. Tokenizer/configuration files are fetched; the Qwen language-model weights are not. |

## Example surgical image

The optional default scene is read from the separate NVIDIA checkout at
`assets/example_data/scenes/example_frame.png`. It is an extracted PNG from
`episode_004148` of the CMR surgical cholecystectomy subset of
[Open-H-Embodiment](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment).

Attribution: **CMR and the Open-H-Embodiment dataset contributors**. The dataset
is maintained by NVIDIA Corporation and stewarded by the Open-H-Embodiment
community and Steering Group. The image is licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
NVIDIA's upstream notice describes extraction from MP4 into PNG; further
preprocessing is unspecified. CholeScope reads the source image without
modifying the stored file; the model preprocesses it for generation.

When sharing captures containing this scene, retain the attribution, source
link, and license link, and label generated continuations and other changes.
Attribution does not imply endorsement. Additional operator-supplied scenes
require their own provenance and permissions.

NVIDIA's separately bundled `trajectory.npy` is an NVIDIA-generated example
under Apache-2.0, not a CMR dataset trajectory. CholeScope uses keyboard-derived
actions rather than that example trajectory.

## Published demo animations

[`assets/cholescope-comparison.gif`](assets/cholescope-comparison.gif) uses the
starting image from CMR cholecystectomy `episode_004148` in
[Open-H-Embodiment](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment/tree/main/Surgical/cmr_surgical/cholecystectomy).
Source attribution: **CMR and the Open-H-Embodiment dataset contributors**,
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Changes: generated A/B continuations with NVIDIA Cosmos3-H-Dreams, resizing,
side-by-side assembly, and annotation. The animation is AI-generated and does
not depict an observed pair of surgical outcomes. Attribution does not imply
endorsement.

## Live interface demo

[`assets/cholescope-live-demo.mp4`](assets/cholescope-live-demo.mp4), its
[animated preview](assets/cholescope-live-preview.gif), and its
[poster](assets/cholescope-live-demo.jpg) show an actual CholeScope browser
session using the same CMR `episode_004148` starting image described above.
Recorded on 1 November 2026 with the full Wan decoder, four-frame chunks,
832 × 480 model output, and seed `42` for both branches.
Attribution: **CMR and the Open-H-Embodiment dataset contributors**.
[Source dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment/tree/main/Surgical/cmr_surgical/cholecystectomy),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Built on NVIDIA Cosmos.

Changes: AI-generated continuations, screen recording, a visible cursor,
two-times playback speed, framing, and chapter captions. The full video is
one continuous 120-second take presented in 60 seconds; setup and model
warmup occur before the take. No generated frames are substituted or
interpolated. The GIF is a resized six-second excerpt and the poster is a
still from the full video. This is a research demonstration, not a clinically
validated simulator or an observed comparison of surgical outcomes.
Attribution does not imply endorsement.

To reproduce the presentation from your own 1600 × 900 browser capture,
install FFmpeg and the project's Python dependencies, then run:

```bash
uv run python scripts/render_readme_demo.py /path/to/capture.webm --start 0
```

`--start` is the source timestamp at which the continuous two-minute take
begins. The renderer adds captions for the demonstrated recording sequence;
adapt their timing if your interaction sequence differs. It produces a
60-second 1920 × 1080 H.264 MP4 below 10 MB, a still preview, and a short GIF.
It does not generate surgical footage. Raw captures, generated branch
videos, and action exports remain local and are not added to the repository.

## Runtime dependencies and provenance

Direct application dependencies are [aiohttp](https://github.com/aio-libs/aiohttp)
(Apache-2.0/MIT), [NumPy](https://numpy.org/) (BSD and bundled-component terms),
and [Pillow](https://python-pillow.org/) (MIT-CMU and bundled-component terms).
The GPU environment adds PyTorch, CUDA components, and other dependencies
documented in NVIDIA's upstream `THIRD-PARTY-NOTICES` and lockfile. They retain
their distribution-specific licenses.

MP4 export invokes a separately installed FFmpeg executable with libx264.
[FFmpeg's build configuration](https://ffmpeg.org/legal.html) determines its
applicable license; x264 is GPL-2.0-or-later or available under a separate
commercial license. CholeScope does not bundle an FFmpeg executable.

The pretrained checkpoint uses Open-H-Embodiment during initialization, while
its final fine-tuning and distillation use proprietary CMR clinical recordings,
as described in the NVIDIA model card. This project does not claim to release
that data, reproduce the training, or validate clinical or physical accuracy.
