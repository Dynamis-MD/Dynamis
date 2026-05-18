import os
import sys
from ovito.io import import_file
from ovito.vis import Viewport, OpenGLRenderer
from ovito.modifiers import SmoothTrajectoryModifier

"""
USAGE: python3 render.py <path_to_xyz_file.xyz> <frame_mod>
frame_mod is optional, it defaults to rendering every frame
if frame_mod is greater than 0, it linearly interpolates with frame_mod extra frames
if frame_mod is less than 0, it renders every frame_mod + 1 frame
"""

if len(sys.argv) < 2:
    print("Error: Please provide the path to the trajectory file.")
    print("Usage: python render.py <path_to_file.xyz>")
    sys.exit(1)

input_file = sys.argv[1]

# Check if the file actually exists before running everything else
if not os.path.exists(input_file):
    print(f"Error: The file '{input_file}' does not exist.")
    sys.exit(1)

output_file = f"{os.path.splitext(os.path.basename(input_file))[0]}.mp4"

# check for extra/less frames
playback_ratio = "1:1"
if len(sys.argv) == 3:
    frames = int(sys.argv[2])
    if frames < 0:
        playback_ratio = f"{-frames + 1}:1"
    elif frames > 0:
        playback_ratio = f"1:{frames + 1}"

# Load trajectory
pipeline = import_file(input_file)
pipeline.source.playback_ratio = playback_ratio
pipeline.add_to_scene()

vp = Viewport()
vp.type = Viewport.Type.Perspective
vp.camera_dir = (-1, -1, -1)
vp.zoom_all()

# Append the custom function to the pipeline modifiers
pipeline.modifiers.append(SmoothTrajectoryModifier())

# Total frames desired in the final video
print(f"Starting render of {pipeline.num_frames} frames...")

vp.render_anim(filename=output_file,
               size=(1920, 1080),
               fps=30,
               background=(1.0, 1.0, 1.0),
               renderer=OpenGLRenderer())

print("Rendering complete!")
