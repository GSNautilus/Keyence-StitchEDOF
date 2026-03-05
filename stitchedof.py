# stitchedof.py 20260305_01
# Garrick Salois
# Script to perform tile stitching from brightfield Keyence images and then perform extended depth-of-field
# Takes tiled image sequence and the .gci metadata file from Keyence as input and outputs single stitched edof image
#
# Requirements: numpy, tifffile, scipy, tqdm, torch
#
# Command examples:
#   python stitchedof.py data
#   python stitchedof.py data custom_output.tif
#   python stitchedof.py data /tmp/out/final.tif --workers 4 --overlap 0.3
#   python stitchedof.py data --tilestitch
#
# CLI arguments:
#   input_dir   Required. Directory containing tile TIFFs and exactly one .gci file.
#   output      Optional. Output TIFF path or filename. If omitted:
#               <input_dir>/<input_dir_name>_stitched.tif
#
# CLI options:
#   --patch_size N    EDOF local-statistics window size (default: 7)
#   --alpha A         EDOF score mixing factor in [0,1] (default: 0.7)
#   --workers N       Number of multiprocessing workers (default: CPU count)
#   --edofsave BOOL   Save per-tile EDOF TIFFs: true/false (default: false)
#   --overlap F       Fractional tile overlap for alignment/blending (default: 0.3)
#   --tilestitch      Use legacy single-tile row anchoring instead of default row-stitch mode

from pathlib import Path
import numpy as np
import tifffile
import torch
import zipfile
import xml.etree.ElementTree as ET
from scipy.ndimage import uniform_filter
from multiprocessing import Pool, cpu_count
import re
import gc
import argparse
from tqdm import tqdm


def edof_from_stack(stack, patch_size=5, alpha=0.6):
    z, h, w = stack.shape
    stack = stack.astype(np.float32)
    stack_min = stack.min()
    stack_ptp = np.ptp(stack) + 1e-6

    edof = np.zeros((h, w), dtype=stack.dtype)
    best_scores = np.full((h, w), -np.inf, dtype=np.float32)

    for i in range(z):
        img = (stack[i] - stack_min) / stack_ptp
        mean = uniform_filter(img, size=patch_size)
        mean_sq = uniform_filter(img ** 2, size=patch_size)
        local_var = mean_sq - mean ** 2
        brightness_score = 1.0 - img
        score = alpha * local_var + (1 - alpha) * brightness_score

        update_mask = score > best_scores
        edof[update_mask] = stack[i][update_mask]
        best_scores[update_mask] = score[update_mask]

    return edof


def edof_worker(args):
    tile_id, slices, config = args
    patch_size = config["patch_size"]
    alpha = config["alpha"]
    save = config["save"]
    output_dir = Path(config["output_dir"])

    slices_sorted = sorted(slices, key=lambda x: x[0])
    stack = np.stack([tifffile.imread(f) for (_, f) in slices_sorted], axis=0)

    edof = edof_from_stack(stack, patch_size, alpha).astype(np.uint16)

    if save:
        out_path = output_dir / f"tile_{tile_id:05d}_edof.tif"
        tifffile.imwrite(out_path, edof)

    del stack, slices_sorted
    gc.collect()
    return tile_id, edof


def extract_tile_grid(gci_path):
    try:
        with zipfile.ZipFile(gci_path, 'r') as z:
            with z.open('GroupFileProperty/ImageJoint/properties.xml') as xml_file:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                rows = int(root.find('.//Row').text)
                cols = int(root.find('.//Column').text)
                if rows < 1 or cols < 1:
                    print("Warning: Parsed 0 rows or columns — defaulting to 1x1 grid.")
                    return 1, 1
                return rows, cols
    except Exception as e:
        print(f"Warning: Failed to parse tile grid from GCI file: {e}")
        print("Defaulting to 1x1 tile grid.")
        return 1, 1


def phase_correlation_gpu(ref_crop, mov_crop, device=None):
    """
    Returns shift as [dx, dy] (how much mov should shift to align to ref).
    """
    H = min(ref_crop.shape[0], mov_crop.shape[0])
    W = min(ref_crop.shape[1], mov_crop.shape[1])
    ref_crop = ref_crop[:H, :W]
    mov_crop = mov_crop[:H, :W]

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ref = torch.from_numpy(ref_crop.copy()).to(dtype=torch.float32, device=device)
    mov = torch.from_numpy(mov_crop.copy()).to(dtype=torch.float32, device=device)

    R = torch.fft.fftn(ref)
    M = torch.fft.fftn(mov)
    eps = 1e-8
    cross_power = (R * torch.conj(M)) / (torch.abs(R * torch.conj(M)) + eps)
    corr = torch.fft.ifftn(cross_power).real

    max_pos = torch.argmax(corr)
    peak = np.unravel_index(max_pos.cpu().item(), corr.shape)  # (py, px)
    peak = np.array(peak, dtype=np.float32)
    shape = np.array(corr.shape, dtype=np.float32)
    mid = shape // 2
    peak[peak > mid] -= shape[peak > mid]

    sz, sx = int(peak[0]), int(peak[1])
    if 1 <= sz <= shape[0] - 2 and 1 <= sx <= shape[1] - 2:
        patch = corr[sz - 1:sz + 2, sx - 1:sx + 2].cpu().numpy()

        def fit_1d(p):
            denom = 2 * p[1] - p[0] - p[2]
            return 0.0 if denom == 0 else 0.5 * (p[0] - p[2]) / denom

        dy = fit_1d(patch[:, 1])
        dx = fit_1d(patch[1, :])

        subpixel_shift = peak + np.array([dy, dx], dtype=np.float32)
        return subpixel_shift[::-1]  # [dx, dy]
    else:
        return peak[::-1]  # [dx, dy]


def stitch_tiles(tile_images, rows, cols, overlap_frac=0.3, row_stitch=True):
    """
    Build global tile positions from serpentine tile IDs.
    - row_stitch=True: row-to-row alignment uses full stitched overlap strips.
    - row_stitch=False: legacy mode anchors each row using one tile pair.
    """

    tile_h, tile_w = next(iter(tile_images.values())).shape
    nominal_step = np.array([tile_w * (1 - overlap_frac), tile_h * (1 - overlap_frac)], dtype=np.float32)

    ov_w = max(1, int(tile_w * overlap_frac))
    ov_h = max(1, int(tile_h * overlap_frac))

    # Keyence tile IDs are serpentine in physical space.
    def tile_id_at(row, col):
        if row % 2 == 0:
            return row * cols + (col + 1)
        else:
            return (row + 1) * cols - col

    positions = {}
    corr_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_label = "CUDA" if corr_device == "cuda" else "CPU"
    mode_label = "ROWSTITCH" if row_stitch else "TILESTITCH"
    print(f"Stitching mode: {mode_label} | Device: {device_label}")

    anchor_col = cols - 1

    def chain_row_left_to_right(row):
        # Build per-row positions using horizontal overlaps only.
        row_positions = {tile_id_at(row, 0): np.array([0.0, 0.0], dtype=np.float32)}
        for col in range(1, cols):
            idx = tile_id_at(row, col)
            left_idx = tile_id_at(row, col - 1)
            ref = tile_images[left_idx]
            mov = tile_images[idx]
            ref_crop = ref[:, -ov_w:]
            mov_crop = mov[:, :ov_w]
            shift = phase_correlation_gpu(ref_crop, mov_crop, device=corr_device)
            offset = np.array([nominal_step[0], 0.0], dtype=np.float32) + shift
            row_positions[idx] = row_positions[left_idx] + offset
        return row_positions

    def render_row_strip(row_positions, row, which):
        # Render only the top or bottom overlap strip of a stitched row.
        row_ids = [tile_id_at(row, c) for c in range(cols)]
        row_coords = np.stack([row_positions[i] for i in row_ids])
        min_coords = row_coords.min(axis=0)
        norm_positions = {i: row_positions[i] - min_coords for i in row_ids}
        max_coords = np.stack([norm_positions[i] for i in row_ids]).max(axis=0)

        strip_h = ov_h
        canvas_w = int(np.ceil(max_coords[0] + tile_w))
        canvas_h = int(np.ceil(max_coords[1] + strip_h))
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for idx in row_ids:
            x, y = np.round(norm_positions[idx]).astype(int)
            tile = tile_images[idx].astype(np.float32)
            strip = tile[:strip_h, :] if which == "top" else tile[-strip_h:, :]
            h, w = strip.shape
            canvas[y:y + h, x:x + w] += strip
            weight[y:y + h, x:x + w] += 1.0

        nz = weight > 0
        canvas[nz] /= weight[nz]
        return canvas, min_coords

    for row in tqdm(range(rows), desc="Computing positions"):
        if row_stitch:
            current_local = chain_row_left_to_right(row)
            if row == 0:
                positions.update(current_local)
            else:
                above_row_positions = {tile_id_at(row - 1, c): positions[tile_id_at(row - 1, c)] for c in range(cols)}
                ref_strip, above_min = render_row_strip(above_row_positions, row - 1, "bottom")
                mov_strip, curr_min = render_row_strip(current_local, row, "top")
                row_shift = phase_correlation_gpu(ref_strip, mov_strip, device=corr_device)
                # Apply nominal row step plus phase-correlation residual.
                row_offset = (
                    above_min
                    - curr_min
                    + np.array([0.0, nominal_step[1]], dtype=np.float32)
                    + row_shift
                )
                for idx, local_pos in current_local.items():
                    positions[idx] = local_pos + row_offset
        else:
            # Legacy mode: anchor each row vertically with one tile pair.
            idx_anchor = tile_id_at(row, anchor_col)

            if row == 0:
                # Seed row 0 and chain horizontally.
                positions[tile_id_at(0, 0)] = np.array([0.0, 0.0], dtype=np.float32)
                for col in range(1, cols):
                    idx = tile_id_at(0, col)
                    left_idx = tile_id_at(0, col - 1)

                    ref = tile_images[left_idx]
                    mov = tile_images[idx]

                    ref_crop = ref[:, -ov_w:]
                    mov_crop = mov[:, :ov_w]
                    shift = phase_correlation_gpu(ref_crop, mov_crop, device=corr_device)

                    offset = np.array([nominal_step[0], 0.0], dtype=np.float32) + shift
                    positions[idx] = positions[left_idx] + offset
            else:
                above_anchor = tile_id_at(row - 1, anchor_col)

                ref = tile_images[above_anchor]
                mov = tile_images[idx_anchor]

                ref_crop = ref[-ov_h:, :]
                mov_crop = mov[:ov_h, :]
                shift = phase_correlation_gpu(ref_crop, mov_crop, device=corr_device)

                offset = np.array([0.0, nominal_step[1]], dtype=np.float32) + shift
                positions[idx_anchor] = positions[above_anchor] + offset

                # Fill remaining tiles in the row via horizontal chaining.
                for col in range(anchor_col - 1, -1, -1):
                    right_idx = tile_id_at(row, col + 1)
                    idx = tile_id_at(row, col)

                    ref = tile_images[right_idx]
                    mov = tile_images[idx]

                    ref_crop = ref[:, :ov_w]
                    mov_crop = mov[:, -ov_w:]
                    shift = phase_correlation_gpu(ref_crop, mov_crop, device=corr_device)

                    offset = np.array([-nominal_step[0], 0.0], dtype=np.float32) + shift
                    positions[idx] = positions[right_idx] + offset

    # Shift coordinates so the stitched canvas starts at (0, 0).
    all_coords = np.stack(list(positions.values()))
    min_coords = all_coords.min(axis=0)
    for k in positions:
        positions[k] -= min_coords

    max_coords = np.stack(list(positions.values())).max(axis=0)
    canvas_w = int(np.ceil(max_coords[0] + tile_w))
    canvas_h = int(np.ceil(max_coords[1] + tile_h))
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    weight_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for idx, pos in positions.items():
        x, y = np.round(pos).astype(int)
        tile = tile_images[idx].astype(np.float32)
        h, w = tile.shape

        # Linear edge ramps reduce seams in overlap regions.
        weight = np.ones_like(tile, dtype=np.float32)

        ramp_w = int(tile_w * overlap_frac)
        if ramp_w > 0:
            ramp = np.linspace(0, 1, ramp_w, dtype=np.float32)
            weight[:, :ramp_w] *= ramp
            weight[:, -ramp_w:] *= ramp[::-1]

        ramp_h = int(tile_h * overlap_frac)
        if ramp_h > 0:
            ramp = np.linspace(0, 1, ramp_h, dtype=np.float32)
            weight[:ramp_h, :] *= ramp[:, None]
            weight[-ramp_h:, :] *= ramp[::-1][:, None]

        canvas[y:y + h, x:x + w] += tile * weight
        weight_map[y:y + h, x:x + w] += weight

    nonzero = weight_map > 0
    canvas[nonzero] /= weight_map[nonzero]
    return np.clip(canvas, 0, 65535).astype(np.uint16)


def main():
    parser = argparse.ArgumentParser(description="EDOF + Stitching Pipeline")
    parser.add_argument("input_dir", help="Input directory containing image tiles and a single .gci file")
    parser.add_argument("output", nargs="?", default=None,
                        help="Optional output .tif path or filename. Defaults to <input_dir_name>_stitched.tif in input_dir.")

    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--edofsave", type=str, default="false")
    parser.add_argument("--overlap", type=float, default=0.3)
    parser.add_argument("--tilestitch", action="store_true",
                        help="Use legacy single-tile row anchoring instead of default full-row overlap stitching.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    gci_files = sorted(input_dir.glob("*.gci"))
    if len(gci_files) != 1:
        raise ValueError(f"Expected exactly one .gci file in {input_dir}, found {len(gci_files)}")
    gci_file = gci_files[0]

    if args.output is None:
        output_path = input_dir / f"{input_dir.name}_stitched.tif"
    else:
        out_arg = Path(args.output)
        if out_arg.parent == Path("."):
            output_path = input_dir / out_arg
        else:
            output_path = out_arg

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    save_edof = args.edofsave.strip().lower() in {"true", "1", "yes", "y", "on"}
    rows, cols = extract_tile_grid(gci_file)
    print(f"Parsed grid: {rows} rows × {cols} columns")

    # Accept both ..._Z###.tif and ..._Z###_CH#.tif naming patterns.
    file_pattern = re.compile(r"(.+?)_(\d{5})_Z(\d{3})(?:_CH\d+)?\.tif$")
    all_files = sorted(f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() == ".tif")

    tile_dict = {}
    for f in all_files:
        m = file_pattern.match(f.name)
        if m:
            tile_id = int(m.group(2))
            z_index = int(m.group(3))
            tile_dict.setdefault(tile_id, []).append((z_index, f))

    if not tile_dict:
        raise ValueError(f"No matching tile files found in {input_dir}")

    config = {
        "patch_size": args.patch_size,
        "alpha": args.alpha,
        "output_dir": output_dir,
        "save": save_edof,
    }

    tasks = [(tile_id, slices, config) for tile_id, slices in tile_dict.items()]
    with Pool(processes=args.workers or cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(edof_worker, tasks), total=len(tasks), desc="EDOF"))

    tile_images = {tile_id: edof for tile_id, edof in results}

    stitched = stitch_tiles(tile_images, rows, cols, overlap_frac=args.overlap, row_stitch=not args.tilestitch)

    tifffile.imwrite(output_path, stitched)
    print(f"Saved stitched image to: {output_path}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
