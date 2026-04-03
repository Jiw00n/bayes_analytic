#! /bin/bash


export TVM_HOME=/root/work/tvm-ansor
export PYTHONPATH=$TVM_HOME/python
export TVM_LIBRARY_PATH=/root/work/tvm-ansor/build-release


tasks_index=(1490)

for idx in "${tasks_index[@]}"; do
    python measure_programs.py --task-index "$idx"
done