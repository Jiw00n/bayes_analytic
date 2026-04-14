#!/bin/bash

set -euo pipefail

source /root/work/venv/bin/activate
export TVM_HOME=/root/work/tvm-ansor
export PYTHONPATH="$TVM_HOME/python:/root/work/tvm-ansor/gallery/constrained_gen_budget"
export TVM_LIBRARY_PATH="$TVM_HOME/build-release"

if [ "$#" -eq 0 ]; then
  set -- \
    --task-index 1490 \
    --summary-csv /root/work/tvm-ansor/gallery/constrained_gen_budget/results/checkpoint_compare_1490/summary.csv \
    --wandb-project none
fi

python /root/work/tvm-ansor/gallery/constrained_gen_budget/train_runtime_ablations.py "$@"
