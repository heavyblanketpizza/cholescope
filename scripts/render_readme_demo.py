#!/usr/bin/env python3
"""Frame a 120-second real browser capture as a 60-second, 2x README demo.

Requires FFmpeg/FFprobe and Pillow. The source must be a 1600 x 900 capture;
--start removes setup before the continuous two-minute take. No inference
footage is synthesized, substituted, interpolated, or retimed independently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont


CHAPTERS = [
    (0, 3, "One image. Two action sequences.", "Built on NVIDIA Cosmos  /  A real browser session"),
    (3, 7, "Start a live session", "CMR starting image  /  Seed 42  /  Frozen Cosmos3-H-Dreams"),
    (7, 10, "01  Record baseline A", "Each recording begins from the same initial condition."),
    (10, 18.5, "Drag to send an action", "Left grasper  /  Shift for precision  /  Fine keyboard input"),
    (18.5, 21.5, "Save the baseline", "The generated video and the model actions are recorded together."),
    (21.5, 24.5, "02  Reset to the same scene and seed", "Change the action sequence while keeping the starting condition fixed."),
    (24.5, 27.5, "Record alternative B", "A second continuation from the same image."),
    (27.5, 36, "Try the opposing input", "Reverse the grasper command and inspect the generated response."),
    (36, 39.5, "Save the alternative", "Both branches are ready for side-by-side review."),
    (39.5, 50, "03  Compare A and B", "Synchronized playback  /  Same starting image and seed  /  Different inputs"),
    (50, 55, "04  Export the evidence", "MP4 recordings + the actual model-action JSON for each branch"),
    (55, 60, "Explore. Compare. Inspect.", "github.com/heavyblanketpizza/cholescope  /  Research demo, not for clinical use"),
]


def run(*args):
    return subprocess.run(args, check=True, capture_output=True).stdout


def probe(path):
    return json.loads(run("ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)))


def font_paths():
    pairs = [
        ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for regular, bold in pairs:
        if Path(regular).is_file() and Path(bold).is_file():
            return regular, bold
    raise SystemExit("Supply --font and --bold-font pointing to installed TTF fonts.")


def artwork(directory, regular, bold):
    bg = Image.new("RGB", (1920, 1080), "#101e19")
    d = ImageDraw.Draw(bg)
    def text(xy, value, size, color, weight=False):
        d.text(xy, value, font=ImageFont.truetype(bold if weight else regular, size), fill=color)
    text((160, 22), "CholeScope", 34, "#f5f7f0", True)
    text((370, 35), "SURGICAL WORLD-MODEL PLAYGROUND", 15, "#a7bcb0")
    d.rounded_rectangle((1430, 23, 1538, 61), radius=19, fill="#243d31")
    text((1448, 32), "LIVE DEMO", 14, "#c9e6d5", True)
    text((1562, 31), "2× SPEED  /  01:00", 19, "#d8e5dd")
    text((160, 1061), "Scene: CMR / Open-H-Embodiment  ·  CC BY 4.0  ·  AI-generated continuations  ·  Not for clinical use", 13, "#96ac9e")
    d.rectangle((158, 84, 1761, 987), outline="#45584b", width=2)
    bg.save(directory / "frame.png")
    for i, (_, _, title, subtitle) in enumerate(CHAPTERS):
        layer = Image.new("RGBA", (1920, 90))
        d = ImageDraw.Draw(layer)
        d.rectangle((160, 12, 164, 59), fill="#9bd8b3")
        d.text((184, 6), title, font=ImageFont.truetype(bold, 27), fill="#f5f7f0")
        d.text((184, 44), subtitle, font=ImageFont.truetype(regular, 17), fill="#b8c9be")
        layer.save(directory / f"chapter-{i:02}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--font")
    parser.add_argument("--bold-font")
    args = parser.parse_args()
    info = probe(args.capture)
    stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    if (stream["width"], stream["height"]) != (1600, 900):
        parser.error("Capture must be 1600 x 900; preserve browser text at native resolution.")
    if args.start < 0 or float(info["format"]["duration"]) < args.start + 120:
        parser.error("The source must contain a complete 120-second take after --start.")
    regular, bold = (args.font, args.bold_font) if args.font and args.bold_font else font_paths()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "cholescope-live-demo.mp4"
    with tempfile.TemporaryDirectory(prefix="cholescope-demo-") as tmp:
        tmp = Path(tmp)
        artwork(tmp, regular, bold)
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                   "-ss", str(args.start), "-t", "120", "-i", str(args.capture),
                   "-loop", "1", "-framerate", "30", "-i", str(tmp / "frame.png")]
        for i in range(len(CHAPTERS)):
            command.extend(["-loop", "1", "-framerate", "30", "-i", str(tmp / f"chapter-{i:02}.png")])
        # Browser recordings may use full-range YUV. Make conversion explicit:
        # implicit overlay negotiation otherwise clips the light interface.
        source_range = "pc" if stream.get("color_range") == "pc" else "tv"
        filters = [
            f"[0:v]scale=in_range={source_range}:out_range=tv:out_color_matrix=bt709,setpts=(PTS-STARTPTS)/2,fps=30[live]",
            "[1:v]scale=out_range=tv:out_color_matrix=bt709,format=yuv420p[matte]",
            "[matte][live]overlay=160:86:shortest=1[base]",
        ]
        last = "base"
        for i, (start, end, _, _) in enumerate(CHAPTERS):
            out = f"chapter{i}"
            filters.append(f"[{i+2}:v]scale=out_range=tv:out_color_matrix=bt709,format=yuva420p[layer{i}]")
            filters.append(f"[{last}][layer{i}]overlay=0:990:enable='gte(t,{start})*lt(t,{end})'[{out}]")
            last = out
        command.extend(["-filter_complex", ";".join(filters), "-map", f"[{last}]", "-an", "-t", "60",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-maxrate", "1100k", "-bufsize", "2200k",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-map_metadata", "-1",
                        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                        "-metadata", "title=CholeScope — 60-second live demo at 2x",
                        "-metadata", "comment=Built on NVIDIA Cosmos. Scene: CMR and Open-H-Embodiment dataset contributors, episode_004148. Source: https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment/tree/main/Surgical/cmr_surgical/cholecystectomy . License: https://creativecommons.org/licenses/by/4.0/ . Changes: AI-generated continuations, browser capture, 2x playback, framing and captions. Research demonstration, not for clinical use.", str(output)])
        run(*command)
    final = probe(output)
    v = next(s for s in final["streams"] if s["codec_type"] == "video")
    if abs(float(final["format"]["duration"]) - 60) > .05 or v["codec_name"] != "h264":
        raise RuntimeError("Unexpected video duration or codec")
    if int(final["format"]["size"]) >= 10_000_000:
        raise RuntimeError("Demo exceeds the 10 MB GitHub attachment target")
    run("ffmpeg", "-v", "error", "-y", "-ss", "44", "-i", str(output), "-frames:v", "1", "-q:v", "2",
        str(args.output_dir / "cholescope-live-demo.jpg"))
    # Short animated README preview links to the full-resolution MP4.
    run("ffmpeg", "-v", "error", "-y", "-ss", "40", "-t", "6", "-i", str(output),
        "-filter_complex", "fps=8,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
        "-loop", "0", str(args.output_dir / "cholescope-live-preview.gif"))
    print(json.dumps({"duration":final["format"]["duration"], "bytes":final["format"]["size"], "dimensions":[v["width"],v["height"]], "codec":v["codec_name"]}))


if __name__ == "__main__":
    main()
