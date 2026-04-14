#!/bin/bash


# checkpoints=("/root/work/tvm-ansor/gallery/constrained_gen_budget/checkpoints/grid_search/lr0.0005_nce0.1_tau0.3_kl0.002_warm10_order.pt")
# checkpoints=(/root/work/tvm-ansor/gallery/constrained_gen_budget/checkpoints/grid_search/lr*.pt)

checkpoints=(/root/work/tvm-ansor/gallery/constrained_gen_budget/checkpoints_all/1490/grid_search/lr*.pt)
# checkpoints=(/root/work/tvm-ansor/gallery/constrained_gen_budget/old/checkpoints/grid_search/lr*.pt)
        #      "/root/work/tvm-ansor/gallery/constrained_gen_budget/checkpoints/lr0.0005_nce0.2_tau0.2_kl0.002_warm20_no_adaln.pt")

# record_json='/root/work/tvm-ansor/gallery/constrained_gen/data/measured_ansor/1490_([3eda1939e30b947e921f5e1814346365,[1,56,56,128],[6,6,32,128],[1,56,56,32]],cuda).json'
# record_json='/root/work/tvm-ansor/gallery/constrained_gen/data/measured_family_ansor/302_([72858abe65e3185202b62d45a3956c75,[1,8,8,128],[6,6,32,128],[1,8,8,32]],cuda).json'


for checkpoint_path in "${checkpoints[@]}"; do
        echo "================================"
        echo "Evaluating checkpoint: $checkpoint_path"
        python tune_by_latent.py \
                --checkpoint "$checkpoint_path" \
                --best-cost \
                --deterministic
                # --latent-gradient

        python tune_by_latent.py \
                --checkpoint "$checkpoint_path" \
                --best-cost
done

# for seed in 1 2 3; do
#         python tune_by_latent.py \
#                 --checkpoint "$checkpoint_path" \
#                 --record-json "$record_json" \
#                 --output . \
#                 --random \
#                 --seed $seed
#                 # --latent-gradient
# done