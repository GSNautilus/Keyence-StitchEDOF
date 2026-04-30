# stitchedof.py 20260305_01
# Garrick Salois
# Script to perform tile stitching from brightfield Keyence images and then perform extended depth-of-field
# Takes tiled image sequence and the .gci metadata file from Keyence as input and outputs single stitched edof image
#
# Requirements: numpy, tifffile, scipy, tqdm, torch
# Optional: psutil (enables memory-aware default for --workers)
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
#   --workers N       Number of multiprocessing workers. Default: auto-sized
#                     from available RAM and a per-tile working-set estimate
#                     (requires psutil); falls back to min(cpu_count, 4) if
#                     psutil is missing.
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
import os
import argparse
import traceback
from tqdm import tqdm

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


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
    py, px = np.unravel_index(max_pos.cpu().item(), corr.shape)
    Hc, Wc = corr.shape

    # Parabolic sub-pixel fit on the integer peak. Use modular indexing so
    # neighbors wrap across the FFT boundary — this keeps the fit symmetric
    # for shifts near zero (where the old signed-coordinate check skipped it).
    y_prev = (py - 1) % Hc
    y_next = (py + 1) % Hc
    x_prev = (px - 1) % Wc
    x_next = (px + 1) % Wc

    c = corr[py, px].item()
    cu = corr[y_prev, px].item()
    cd = corr[y_next, px].item()
    cl = corr[py, x_prev].item()
    cr = corr[py, x_next].item()

    def fit_1d(a, b, c_):
        denom = 2 * b - a - c_
        return 0.0 if denom == 0 else 0.5 * (a - c_) / denom

    dy = float(np.clip(fit_1d(cu, c, cd), -1.0, 1.0))
    dx = float(np.clip(fit_1d(cl, c, cr), -1.0, 1.0))

    # Convert integer peak to signed shift via FFT wraparound convention,
    # then add the sub-pixel residual.
    peak = np.array([py, px], dtype=np.float32)
    shape = np.array(corr.shape, dtype=np.float32)
    mid = shape // 2
    peak[peak > mid] -= shape[peak > mid]

    subpixel_shift = peak + np.array([dy, dx], dtype=np.float32)
    return subpixel_shift[::-1]  # [dx, dy]


def stitch_tiles(tile_images, rows, cols, overlap_frac=0.3, row_stitch=True):
    """
    Build global tile positions from serpentine tile IDs.
    - row_stitch=True: row-to-row alignment uses full stitched overlap strips.
    - row_stitch=False: legacy mode anchors each row using one tile pair.
    """

    tile_h, tile_w = next(iter(tile_images.values())).shape

    ov_w = max(1, int(tile_w * overlap_frac))
    ov_h = max(1, int(tile_h * overlap_frac))

    # Step must match the integer strip size used when rendering overlap regions;
    # a float step like tile_dim*(1-overlap) drifts by the fractional part each row/col.
    nominal_step = np.array([tile_w - ov_w, tile_h - ov_h], dtype=np.float32)

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
    # +1 in each dimension so bilinear placement has room for its (h+1, w+1) footprint.
    canvas_w = int(np.ceil(max_coords[0] + tile_w)) + 1
    canvas_h = int(np.ceil(max_coords[1] + tile_h)) + 1
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    weight_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for idx, pos in positions.items():
        x_f, y_f = float(pos[0]), float(pos[1])
        x_int = int(np.floor(x_f))
        y_int = int(np.floor(y_f))
        x_frac = x_f - x_int
        y_frac = y_f - y_int

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

        weighted_tile = tile * weight

        # Bilinear sub-pixel placement: distribute each tile pixel across the
        # 4 surrounding canvas pixels by the fractional offset, so we keep the
        # sub-pixel precision that phase correlation recovered.
        w00 = (1.0 - x_frac) * (1.0 - y_frac)
        w10 = x_frac * (1.0 - y_frac)
        w01 = (1.0 - x_frac) * y_frac
        w11 = x_frac * y_frac

        canvas[y_int:y_int + h,     x_int:x_int + w]         += weighted_tile * w00
        canvas[y_int:y_int + h,     x_int + 1:x_int + w + 1] += weighted_tile * w10
        canvas[y_int + 1:y_int + h + 1, x_int:x_int + w]         += weighted_tile * w01
        canvas[y_int + 1:y_int + h + 1, x_int + 1:x_int + w + 1] += weighted_tile * w11

        weight_map[y_int:y_int + h,     x_int:x_int + w]         += weight * w00
        weight_map[y_int:y_int + h,     x_int + 1:x_int + w + 1] += weight * w10
        weight_map[y_int + 1:y_int + h + 1, x_int:x_int + w]         += weight * w01
        weight_map[y_int + 1:y_int + h + 1, x_int + 1:x_int + w + 1] += weight * w11

    nonzero = weight_map > 0
    canvas[nonzero] /= weight_map[nonzero]
    return np.clip(canvas, 0, 65535).astype(np.uint16)


def estimate_per_worker_bytes(tile_dict):
    """
    Estimate peak memory used by one EDOF worker on the largest tile-stack in
    the dataset.

    The peak in edof_from_stack happens during `stack.astype(np.float32)`,
    where the uint16 stack and the new float32 stack briefly coexist:
        uint16_stack + float32_stack = z*h*w*2 + z*h*w*4 = 1.5 x float32_stack
    After that, steady state is ~1 x float32_stack plus a handful of single-
    slice (h*w) intermediates inside the loop, which are negligible compared
    to the stack.

    We model peak as 2.0 x float32_stack — the 1.5x astype overlap rounded up
    for Python/numpy overhead, list-of-slices memory during np.stack, and
    interpreter footprint. The downstream "use 70% of available RAM" budget
    provides additional safety margin.
    """
    if not tile_dict:
        return 0
    # Use the deepest stack and a real tile to read shape/dtype.
    sample_id = max(tile_dict, key=lambda k: len(tile_dict[k]))
    slices = tile_dict[sample_id]
    z = len(slices)
    sample = tifffile.imread(slices[0][1])
    h, w = sample.shape[-2], sample.shape[-1]
    float32_stack = z * h * w * 4
    return int(float32_stack * 2.0)


def available_memory_bytes():
    """Available RAM in bytes, or None if psutil is missing."""
    if not _HAS_PSUTIL:
        return None
    try:
        return psutil.virtual_memory().available
    except Exception:
        return None


def choose_workers(tile_dict, requested):
    """
    Pick a worker count that won't OOM. Honors --workers if given; otherwise
    sizes by available RAM and CPU count. Falls back to a conservative default
    when psutil isn't installed.
    """
    cpu = cpu_count() or 1
    per_worker = estimate_per_worker_bytes(tile_dict)
    avail = available_memory_bytes()

    if requested is not None and requested > 0:
        chosen = requested
        source = "user-specified"
    elif avail is None:
        # No psutil: be conservative. Most boxes survive 4 workers; users with
        # bigger machines can override with --workers.
        chosen = max(1, min(cpu, 4))
        source = "default (psutil not installed; install for memory-aware sizing)"
    else:
        # Use ~70% of available RAM as the budget — leaves headroom for the
        # stitching stage, the OS, and anything else running.
        budget = int(avail * 0.70)
        by_mem = max(1, budget // max(per_worker, 1))
        chosen = max(1, min(cpu, by_mem))
        source = "memory-aware auto"

    print(
        f"Workers: {chosen} ({source}) | "
        f"CPU cores: {cpu} | "
        f"Est. per-worker peak: {per_worker / 1e9:.2f} GB"
        + (f" | Available RAM: {avail / 1e9:.2f} GB" if avail is not None else "")
    )
    return chosen, per_worker


def print_oom_help(workers, per_worker_bytes, exc):
    """Verbose, actionable diagnostic when the EDOF stage runs out of memory."""
    suggested = max(1, workers // 2)
    avail = available_memory_bytes()
    bar = "=" * 72
    print("\n" + bar)
    print("ERROR: The EDOF stage ran out of memory (or a worker process died).")
    print(bar)
    print(f"Underlying error: {type(exc).__name__}: {exc}")
    print(f"Workers used:               {workers}")
    print(f"Estimated peak per worker:  {per_worker_bytes / 1e9:.2f} GB")
    print(f"Estimated total at peak:    {workers * per_worker_bytes / 1e9:.2f} GB")
    if avail is not None:
        print(f"Available RAM right now:    {avail / 1e9:.2f} GB")
    if not _HAS_PSUTIL:
        print("Note: psutil is not installed, so the auto-default could not size")
        print("      itself to your machine. Install it for better defaults:")
        print("          pip install psutil")
    print("")
    print("WHAT HAPPENED")
    print("  Each EDOF worker loads a full Z-stack for one tile and builds")
    print("  several float32 buffers the size of that stack. With many workers")
    print("  in parallel, peak RAM = workers x per-worker working set, which")
    print("  can exceed available memory and crash the run.")
    print("")
    print("HOW TO FIX")
    print(f"  Re-run with fewer workers, e.g.:")
    print(f"      --workers {suggested}")
    print("  If --workers 1 still OOMs, your tile Z-stacks are too large to")
    print("  fit; close other apps or run on a machine with more RAM.")
    print(bar + "\n")


def main():
    parser = argparse.ArgumentParser(description="EDOF + Stitching Pipeline")
    parser.add_argument("input_dir", help="Input directory containing image tiles and a single .gci file")
    parser.add_argument("output", nargs="?", default=None,
                        help="Optional output .tif path or filename. Defaults to <input_dir_name>_stitched.tif in input_dir.")

    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=None,
                        help="Multiprocessing workers. Default: memory-aware "
                             "auto-sizing (uses ~70%% of available RAM, capped "
                             "at CPU count). Requires psutil; falls back to "
                             "min(cpu_count, 4) without it.")
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
    workers, per_worker_bytes = choose_workers(tile_dict, args.workers)
    try:
        with Pool(processes=workers) as pool:
            results = list(tqdm(pool.imap_unordered(edof_worker, tasks), total=len(tasks), desc="EDOF"))
    except (MemoryError, BrokenPipeError, EOFError, OSError) as e:
        print_oom_help(workers, per_worker_bytes, e)
        raise
    except Exception as e:
        # Worker crashes can surface as generic exceptions on Windows; if the
        # message looks memory-related, show the OOM help anyway.
        msg = f"{type(e).__name__}: {e}".lower()
        if any(s in msg for s in ("memory", "alloc", "paging", "0xc00000fd", "killed")):
            print_oom_help(workers, per_worker_bytes, e)
        else:
            traceback.print_exc()
        raise

    tile_images = {tile_id: edof for tile_id, edof in results}

    stitched = stitch_tiles(tile_images, rows, cols, overlap_frac=args.overlap, row_stitch=not args.tilestitch)

    tifffile.imwrite(output_path, stitched)
    print(f"Saved stitched image to: {output_path}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
