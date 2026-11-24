import os
import argparse
import numpy as np
from ovito.data import CutoffNeighborFinder, DataCollection
from ovito.io import import_file
from ovito.vis import Viewport, OpenGLRenderer, TachyonRenderer
from ovito.modifiers import SmoothTrajectoryModifier
from ovito_loader import BinaryReader

"""
USAGE: python3 render.py <path_to_traj_file> [--frame-mod N] [--max-frames N]
                         [--image] [--renderer {opengl,tachyon}]
                         [--hide-water] [--center] [--hide-cell]
                         [--view {diag,px,mx,py,my,pz,mz} ...]
--frame-mod: if > 0, linearly interpolates with N extra frames between each pair;
             if < 0, renders every (|N|+1)th frame
--max-frames: only render the first N frames of the trajectory
--image: render frame 0 to a PNG instead of an MP4 (implied for single-frame inputs)
--renderer: opengl (default, needs GPU/EGL) or tachyon (CPU ray tracer)
--hide-water: delete water molecules, detected geometrically
--center: shift the system so the solute isn't split across periodic boundaries
--hide-cell: don't draw the simulation cell; the camera then frames only the atoms
--view: camera position(s). diag looks in from the (+,+,+) corner; px/mx etc. look
        straight at the +x/-x face (orthographic). Multiple views are only
        supported for still images, one PNG per view with the view in the name.
"""

# Covers O-H (~0.96 A) while excluding H-bond partners (~1.8 A) and the
# O-C/O-P bonds (>1.3 A) that distinguish non-water oxygens.
WATER_OH_CUTOFF = 1.25
CENTER_HIST_BINS = 64

# view name -> (camera direction, up vector, orthographic)
VIEWS = {
    "diag": ((-1, -1, -1), (0, 0, 1), False),
    "px": ((-1, 0, 0), (0, 0, 1), True),
    "mx": ((1, 0, 0), (0, 0, 1), True),
    "py": ((0, -1, 0), (0, 0, 1), True),
    "my": ((0, 1, 0), (0, 0, 1), True),
    # Up can't be parallel to the view direction.
    "pz": ((0, 0, -1), (0, 1, 0), True),
    "mz": ((0, 0, 1), (0, 1, 0), True),
}


def water_mask(data: DataCollection) -> np.ndarray:
    """Marks atoms of molecules whose O has exactly two neighbours, both H.

    Inputs carry only element + position, so water has to be inferred from
    geometry rather than residue names.
    """
    types = data.particles.particle_types
    o_type = types.type_by_name("O")
    h_type = types.type_by_name("H")
    n = data.particles.count
    if o_type is None or h_type is None:
        return np.zeros(n, dtype=bool)

    t = np.asarray(types)
    pairs, _ = CutoffNeighborFinder(WATER_OH_CUTOFF, data).find_all()
    centers, neighbors = pairs[:, 0], pairs[:, 1]
    total = np.bincount(centers, minlength=n)
    h_count = np.bincount(centers[t[neighbors] == h_type.id], minlength=n)

    is_water_o = (t == o_type.id) & (total == 2) & (h_count == 2)
    mask = is_water_o.copy()
    mask[neighbors[is_water_o[centers]]] = True
    return mask


def hide_water(frame: int, data: DataCollection):
    data.particles_.delete_elements(water_mask(data))


def centering_shift(data: DataCollection) -> np.ndarray:
    """Fractional shift per axis that moves the widest solute-free gap onto
    the cell boundary, so the solute ends up contiguous in the middle."""
    cell = np.asarray(data.cell)
    frac = (np.asarray(data.particles.positions) - cell[:, 3]) @ np.linalg.inv(cell[:, :3]).T
    frac = frac[~water_mask(data)] % 1.0

    shift = np.zeros(3)
    for axis in range(3):
        if not data.cell.pbc[axis] or len(frac) == 0:
            continue
        hist, _ = np.histogram(frac[:, axis], bins=CENTER_HIST_BINS, range=(0.0, 1.0))
        empty = np.concatenate([hist, hist]) == 0  # doubled to handle wrap-around runs
        best_len, best_end, run = 0, 0, 0
        for i, e in enumerate(empty):
            run = run + 1 if e else 0
            if best_len < run <= CENTER_HIST_BINS:
                best_len, best_end = run, i
        if best_len == 0:
            continue  # solute spans the whole axis, nothing to fix
        gap_center = (best_end - (best_len - 1) / 2 + 0.5) / CENTER_HIST_BINS
        shift[axis] = -gap_center % 1.0
    return shift


def make_center_modifier(shift: np.ndarray):
    def center(frame: int, data: DataCollection):
        cell = np.asarray(data.cell)
        h = cell[:, :3]
        frac = (np.asarray(data.particles.positions) - cell[:, 3]) @ np.linalg.inv(h).T
        frac += shift
        pbc = np.asarray(data.cell.pbc, dtype=bool)
        frac[:, pbc] %= 1.0
        data.particles_.positions_[...] = frac @ h.T + cell[:, 3]

    return center

parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("--frame-mod", type=int, default=0)
parser.add_argument("--max-frames", type=int, default=None)
parser.add_argument("--image", action="store_true")
parser.add_argument("--renderer", choices=("opengl", "tachyon"), default="opengl")
parser.add_argument("--hide-water", action="store_true")
parser.add_argument("--center", action="store_true")
parser.add_argument("--hide-cell", action="store_true")
parser.add_argument("--view", nargs="+", choices=tuple(VIEWS), default=["diag"])
args = parser.parse_args()

input_file = args.input_file

if not os.path.exists(input_file):
    print(f"Error: The file '{input_file}' does not exist.")
    import sys

    sys.exit(1)

base, ext = os.path.splitext(os.path.basename(input_file))

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
if args.hide_cell:
    pipeline.source.data.cell.vis.enabled = False
pipeline.add_to_scene()

# Shift is fixed from frame 0 so the system doesn't jump between frames.
if args.center:
    pipeline.modifiers.append(make_center_modifier(centering_shift(pipeline.source.compute(0))))
if args.hide_water:
    pipeline.modifiers.append(hide_water)

# Append the custom function to the pipeline modifiers
pipeline.modifiers.append(SmoothTrajectoryModifier())

def make_viewport(view: str) -> Viewport:
    direction, up, ortho = VIEWS[view]
    vp = Viewport()
    vp.type = Viewport.Type.Ortho if ortho else Viewport.Type.Perspective
    vp.camera_dir = direction
    vp.camera_up = up
    vp.zoom_all()
    return vp


renderer = TachyonRenderer() if args.renderer == "tachyon" else OpenGLRenderer()
size = (1920, 1080)
background = (1.0, 1.0, 1.0)

# Default invocation keeps the original output name.
suffix_parts = []
if args.hide_cell:
    suffix_parts.append("nocell")


def output_name(view: str, ext: str) -> str:
    parts = list(suffix_parts)
    if args.view != ["diag"]:
        parts.append(f"view-{view}")
    return "_".join([base, *parts]) + ext


if args.image or pipeline.num_frames == 1:
    for view in args.view:
        output_file = output_name(view, ".png")
        print(f"Rendering frame 0 to {output_file}...")
        make_viewport(view).render_image(
            filename=output_file,
            size=size,
            frame=0,
            background=background,
            renderer=renderer,
        )
    print("Rendering complete!")
    raise SystemExit(0)

if len(args.view) != 1:
    parser.error("animations support a single --view")

output_file = output_name(args.view[0], ".mp4")
end_frame = pipeline.num_frames - 1
if args.max_frames is not None:
    end_frame = min(args.max_frames - 1, end_frame)

print(f"Starting render of {end_frame + 1} frames...")

make_viewport(args.view[0]).render_anim(
    filename=output_file,
    size=size,
    fps=30,
    range=(0, end_frame),
    background=background,
    renderer=renderer,
)

print("Rendering complete!")
