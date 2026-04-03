#!/bin/bash

python tune.py \
        --checkpoint /root/work/tvm-ansor/gallery/constrained_gen/latent_param_model/checkpoints_latent_param/last_04031104_acc_0.71.pt \
        --record-json '/root/work/tvm-ansor/gallery/constrained_gen/data/to_measure_ansor/1490_ansor_([3eda1939e30b947e921f5e1814346365,[1,56,56,128],[6,6,32,128],[1,56,56,32]],cuda).json' \
        --output /root/work/tvm-ansor/gallery/constrained_gen/data/tuned
        # --latent-gradient