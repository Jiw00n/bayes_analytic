#!/bin/bash


checkpoint_path="/root/work/tvm-ansor/gallery/constrained_gen_budget/checkpoints/grid_search/lr0.0005_nce0.1_tau0.3_kl0.002_warm10_order.pt"
record_json='/root/work/tvm-ansor/gallery/constrained_gen/data/measured_ansor/1490_([3eda1939e30b947e921f5e1814346365,[1,56,56,128],[6,6,32,128],[1,56,56,32]],cuda).json'


python tune_by_latent.py \
        --checkpoint "$checkpoint_path" \
        --record-json "$record_json" \
        --output . \
        --best-cost \
        --deterministic
        # --latent-gradient

python tune_by_latent.py \
        --checkpoint "$checkpoint_path" \
        --record-json "$record_json" \
        --output . \
        --best-cost

for seed in 1 2 3; do
        python tune_by_latent.py \
                --checkpoint "$checkpoint_path" \
                --record-json "$record_json" \
                --output . \
                --random \
                --seed $seed
                # --latent-gradient
done