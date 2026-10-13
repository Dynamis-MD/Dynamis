import os
import argparse
from ovito.io import import_file
from ovito.vis import Viewport, OpenGLRenderer
from ovito.modifiers import SmoothTrajectoryModifier
from ovito_loader import BinaryReader

"""
USAGE: python3 render.py <path_to_traj_file> [--frame-mod N] [--max-frames N]
--frame-mod: if > 0, linearly interpolates with N extra frames between each pair;
             if < 0, renders every (|N|+1)th frame
--max-frames: only render the first N frames of the trajectory
"""

parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("--frame-mod", type=int, default=0)
parser.add_argument("--max-frames", type=int, default=None)
args = parser.parse_args()

input_file = args.input_file

if not os.path.exists(input_file):
    print(f"Error: The file '{input_file}' does not exist.")
    import sys; sys.exit(1)

base, ext = os.path.splitext(os.path.basename(input_file))
output_file = f"{base}.mp4"

playback_ratio = "1:1"
if args.frame_mod < 0:
    playback_ratio = f"{-args.frame_mod + 1}:1"
elif args.frame_mod > 0:
    playback_ratio = f"1:{args.frame_mod + 1}"

# Load trajectory
if ext == ".dynamis":
    pipeline = import_file(input_file, input_format=BinaryReader)
else:
    pipeline = import_file(input_file)
pipeline.source.playback_ratio = playback_ratio
pipeline.add_to_scene()

vp = Viewport()
vp.type = Viewport.Type.Perspective
vp.camera_dir = (-1, -1, -1)
vp.zoom_all()

# Append the custom function to the pipeline modifiers
pipeline.modifiers.append(SmoothTrajectoryModifier())

end_frame = pipeline.num_frames - 1
if args.max_frames is not None:
    end_frame = min(args.max_frames - 1, end_frame)

print(f"Starting render of {end_frame + 1} frames...")

vp.render_anim(filename=output_file,
               size=(1920, 1080),
               fps=30,
               range=(0, end_frame),
               background=(1.0, 1.0, 1.0),
               renderer=OpenGLRenderer())

print("Rendering complete!")
