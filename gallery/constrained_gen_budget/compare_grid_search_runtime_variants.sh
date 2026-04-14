#!/bin/bash

set -euo pipefail

ROOT_DIR="/root/work/tvm-ansor/gallery/constrained_gen_budget"
BASE_SCRIPT="$ROOT_DIR/compare_grid_search_checkpoints.sh"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-$ROOT_DIR/results/checkpoint_compare_1490_variants}"
VARIANTS="${VARIANTS:-current default_constraints}"

mkdir -p "$OUTPUT_ROOT_BASE"

for variant in $VARIANTS; do
    echo "================================"
    echo "Variant: $variant"

    CGB_USE_DEFAULT_ENABLED_CONSTRAINTS=0

    case "$variant" in
        current)
            ;;
        default_constraints)
            CGB_USE_DEFAULT_ENABLED_CONSTRAINTS=1
            ;;
        *)
            echo "Unknown variant: $variant"
            exit 1
            ;;
    esac

    OUTPUT_ROOT="$OUTPUT_ROOT_BASE/$variant" \
    CGB_USE_DEFAULT_ENABLED_CONSTRAINTS="$CGB_USE_DEFAULT_ENABLED_CONSTRAINTS" \
    "$BASE_SCRIPT"
done

echo "================================"
echo "All variants finished."
echo "  output root: $OUTPUT_ROOT_BASE"
