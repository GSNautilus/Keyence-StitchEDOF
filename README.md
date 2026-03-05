# Keyence-StitchEDOF

`stitchedof.py` performs two steps for tiled Keyence datasets:
1. Per-tile extended depth-of-field (EDOF) collapse from Z-stacks.
2. Global tile stitching into one output TIFF.

The script expects an input folder containing:
- Tile TIFF files named like `..._00001_Z001.tif` or `..._00001_Z001_CH4.tif`
- Exactly one `.gci` file (used to read row/column grid size)

## Requirements

- Python packages: `numpy`, `tifffile`, `scipy`, `tqdm`, `torch`
- CUDA-enabled PyTorch is optional but recommended for faster phase-correlation stitching.

PyTorch install:
- https://pytorch.org/get-started/locally/

## Usage

```bash
python stitchedof.py <input_dir> [output]
```

- `input_dir`: required directory containing TIFF tiles and one `.gci`
- `output`: optional output TIFF path or filename
  - If omitted, default output is: `<input_dir>/<input_dir_name>_stitched.tif`

Examples:

```bash
python stitchedof.py data
python stitchedof.py data custom_output.tif
python stitchedof.py data /tmp/out/final.tif --workers 4 --overlap 0.3
python stitchedof.py data --tilestitch
```

## Options

- `--patch_size N` EDOF local-statistics window size (default: `7`)
- `--alpha A` EDOF score mixing factor in `[0,1]` (default: `0.7`)
- `--workers N` multiprocessing workers for EDOF tiles (default: CPU count)
- `--edofsave BOOL` save per-tile EDOF TIFFs (`true/false`, default: `false`)
- `--overlap F` fractional tile overlap for alignment/blending (default: `0.3`)
- `--tilestitch` use legacy single-tile row anchoring instead of default full-row overlap stitching

## Notes

- Default stitching mode is row-based (`ROWSTITCH`), which aligns rows using full overlap strips.
- `--tilestitch` switches to legacy behavior using one anchor tile pair per row.
- Runtime console output reports stitching mode and device (`CPU` or `CUDA`).
