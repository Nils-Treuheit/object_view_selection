#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Array of target dataset directories
PATHS=(
"/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/intresting_objects/thermos_bottle"
"/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/intresting_objects/elephant"
"/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/intresting_objects/glass_bottle"
"/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/intresting_objects/keys"
"/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/intresting_objects/nit_cube"
"/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/intresting_objects/ovgu_mug"
"/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/intresting_objects/ovgu_mug_dark"
"/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/intresting_objects/sun_screen"
"/mnt/HDD1/Project_Code/nit_object_onboarding/workspace/intresting_objects/wallet"
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
