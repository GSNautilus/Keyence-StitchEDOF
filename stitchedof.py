#StitchEDOF.py 
#202511
#Garrick Salois
#Script to perform tile stitching from brightfield Keyence images and then perform extended depth-of-field
#Takes tiled image sequence and the .gci metadata file from Keyence as input and outputs single stitched edof image
#
#Requirements: numpy, tifffile, scipy, tqdm, torch
#Torch ideally should be installed for a CUDA capable GPU and for your specific version of CUDA, i.e.:
#https://pytorch.org/get-started/locally/
#
#Typical command line 
#python GolgiParse.py <input_dir> <output_dir> <gci_filepath> <export_stack_filename>
#
#--workers arg can be adjusted if you get out-of-memory errors, should be less than # of system cores
#--alpha and --patch_size can be adjusted from defaults to adjust the extended depth of field algorithm parameters
#
#Example:
#python GolgiParse.py \\smdnas02\example_folder\M_KO_47\47 D:/Garrick/Keyence/Golgi2025/MKO/ \\smdnas02\example_folder\47\M_KO.gci mko_47.tif --workers 26

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

def phase_correlation_gpu(ref_crop, mov_crop):
    H = min(ref_crop.shape[0], mov_crop.shape[0])
    W = min(ref_crop.shape[1], mov_crop.shape[1])
    ref_crop = ref_crop[:H, :W]
    mov_crop = mov_crop[:H, :W]

    ref = torch.from_numpy(ref_crop.copy()).to(dtype=torch.float32, device='cuda')
    mov = torch.from_numpy(mov_crop.copy()).to(dtype=torch.float32, device='cuda')

    R = torch.fft.fftn(ref)
    M = torch.fft.fftn(mov)
    eps = 1e-8
    cross_power = (R * torch.conj(M)) / (torch.abs(R * torch.conj(M)) + eps)
    corr = torch.fft.ifftn(cross_power).real

    max_pos = torch.argmax(corr)
    peak = np.unravel_index(max_pos.cpu().item(), corr.shape)
    peak = np.array(peak, dtype=np.float32)
    shape = np.array(corr.shape)
    mid = shape // 2
    peak[peak > mid] -= shape[peak > mid]

    sz, sx = int(peak[0]), int(peak[1])
    if 1 <= sz <= shape[0] - 2 and 1 <= sx <= shape[1] - 2:
        patch = corr[sz-1:sz+2, sx-1:sx+2].cpu().numpy()

        def fit_1d(p):
            denom = 2 * p[1] - p[0] - p[2]
            return 0.0 if denom == 0 else 0.5 * (p[0] - p[2]) / denom

        dy = fit_1d(patch[:, 1])
        dx = fit_1d(patch[1, :])
        subpixel_shift = peak + np.array([dy, dx], dtype=np.float32)
        return subpixel_shift[::-1]  # [dy, dx]
    else:
        return peak[::-1]

def stitch_tiles(tile_images, rows, cols, overlap_frac=0.3):
    tile_shape = next(iter(tile_images.values())).shape
    tile_h, tile_w = tile_shape
    nominal_step = np.array([tile_w * (1 - overlap_frac), tile_h * (1 - overlap_frac)])

    positions = {}
    for row in tqdm(range(rows), desc="Computing positions"):
        for col in range(cols):
            col_in_snake = col if row % 2 == 0 else cols - col - 1
            idx = row * cols + col_in_snake + 1
            is_first_tile_in_row = (col == 0)

            if row == 0 and is_first_tile_in_row:
                positions[idx] = np.array([0.0, 0.0], dtype=np.float32)
                continue

            if is_first_tile_in_row:
                above_row = row - 1
                above_col_in_snake = col if above_row % 2 == 0 else cols - col - 1
                above_idx = above_row * cols + above_col_in_snake + 1
                ref = tile_images[above_idx]
                mov = tile_images[idx]
                ref_crop = ref[-int(tile_h * overlap_frac):, :]
                mov_crop = mov[:int(tile_h * overlap_frac), :]
                shift = phase_correlation_gpu(ref_crop, mov_crop)
                offset = np.array([0, nominal_step[1]]) + shift
                positions[idx] = positions[above_idx] + offset
            else:
                if row % 2 == 0:
                    neighbor_idx = row * cols + col_in_snake
                else:
                    neighbor_idx = row * cols + col_in_snake + 2
                ref = tile_images[neighbor_idx]
                mov = tile_images[idx]
                ref_crop = ref[:, -int(tile_w * overlap_frac):]
                mov_crop = mov[:, :int(tile_w * overlap_frac)]
                shift = phase_correlation_gpu(ref_crop, mov_crop)
                offset = np.array([nominal_step[0], 0]) + shift
                positions[idx] = positions[neighbor_idx] + offset

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

# ----------------- MAIN -----------------
def main():
    parser = argparse.ArgumentParser(description="EDOF + Stitching Pipeline")
    parser.add_argument("input_dir", help="Input directory with *_Z???_CH4.tif")
    parser.add_argument("output_dir", help="Directory to save EDOF tiles and stitched image")
    parser.add_argument("gci_file", help="Path to .gci metadata ZIP file")
    parser.add_argument("filename", type=str, default="stitched.tif", help="Filename for the stitched image (e.g., final_stitch.tif). If omitted, defaults to 'stitched.tif'.")

    parser.add_argument("--patch_size", type=int, default=7, help="Parameter that adjusts size of EDOF filter; smaller numbers take more time but may also capture finer detail (including extraneous noise if too small), Default=7")
    parser.add_argument("--alpha", type=float, default=0.7, help="Scale from 0-1. Adjusts the weighting for intensity vs. variance in the EDOF algorithm. Increase this number to apply more weighting to variance vs. intensity. Default=0.7")
    parser.add_argument("--workers", type=int, default=None, help="Number of simultaneous jobs to run. If none set, will utilize all cpu cores by default. Default=None")
    parser.add_argument("--edofsave", type=str, default="false", help="Option to save individual tiles after edof processing. Default=False")
    parser.add_argument("--overlap", type=float, default=0.3, help="The tile overlap fraction. Default=0.3")
    args = parser.parse_args()

    save_edof = args.edofsave.lower() == "true"
    rows, cols = extract_tile_grid(args.gci_file)
    print(f"Parsed grid: {rows} rows × {cols} columns")

    file_pattern = re.compile(r"(.+?)_(\d{5})_Z(\d{3})_CH4\.tif")
    all_files = sorted(Path(args.input_dir).glob("*_CH4.tif"))

    tile_dict = {}
    for f in all_files:
        m = file_pattern.match(f.name)
        if m:
            tile_id = int(m.group(2))
            z_index = int(m.group(3))
            tile_dict.setdefault(tile_id, []).append((z_index, f))

    config = {
        "patch_size": args.patch_size,
        "alpha": args.alpha,
        "output_dir": args.output_dir,
        "save": save_edof,
    }

    tasks = [(tile_id, slices, config) for tile_id, slices in tile_dict.items()]
    with Pool(processes=args.workers or cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(edof_worker, tasks), total=len(tasks), desc="EDOF"))

    tile_images = {tile_id: edof for tile_id, edof in results}
    stitched = stitch_tiles(tile_images, rows, cols, overlap_frac=args.overlap)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output_dir exists

    output_path = output_dir / args.filename
    tifffile.imwrite(output_path, stitched)
    print(f"Saved stitched image to: {output_path}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()

