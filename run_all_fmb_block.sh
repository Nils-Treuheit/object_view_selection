#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Array of target dataset directories
PATHS=(
    "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/fmb_blocks/01_circ_cyl"
    "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/fmb_blocks/02_oval_cyl"
    "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/fmb_blocks/03_cuboid"
    "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/fmb_blocks/04_unotch"
    "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/fmb_blocks/05_hexagon"
    "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/fmb_blocks/06_star"
    "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/fmb_blocks/07_twin_square"
    "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/fmb_blocks/08_round_square"
    "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/fmb_blocks/09_triprong"
    "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/09_triprong_old"
    "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/nit_cube"
    "/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/bottle"
)

# Loop over each path
for DATA_ROOT in "${PATHS[@]}"; do
    # Remove trailing slash if present to ensure clean output_dir suffix
    DATA_ROOT="${DATA_ROOT%/}"
    
    OUTPUT_DIR="${DATA_ROOT}_cleaned"
    
    echo "=================================================================="
    echo "Processing: ${DATA_ROOT}"
    echo "Output Dir: ${OUTPUT_DIR}"
    echo "=================================================================="
    
    python run.py \
        --data_root "${DATA_ROOT}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_views 10 \
        --plot \
        --debug \
        --selector top_kmeans_xnn \
        --kmeans_init farthest \
        --kmeans_xnn_k 10

    echo ""
done

echo "All paths processed successfully!"
