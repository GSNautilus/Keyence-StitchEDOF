Keyence-StitchEDOF
Script to perform tile stitching from brightfield Keyence images and then perform extended depth-of-field. Takes tiled image sequence and the .gci metadata file from Keyence as input and outputs single stitched edof image

StitchEDOF.py 
202511
Garrick Salois
Script to perform tile stitching from brightfield Keyence images and then perform extended depth-of-field
Takes tiled image sequence and the .gci metadata file from Keyence as input and outputs single stitched edof image

Requirements: numpy, tifffile, scipy, tqdm, torch
Torch ideally should be installed for a CUDA capable GPU and for your specific version of CUDA, i.e.:
https://pytorch.org/get-started/locally/

Typical command line 
python GolgiParse.py <input_dir> <output_dir> <gci_filepath> <export_stack_filename>

--workers arg can be adjusted if you get out-of-memory errors, should be less than # of system cores

--alpha and --patch_size can be adjusted from defaults to adjust the extended depth of field algorithm parameters
