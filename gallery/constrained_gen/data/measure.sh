#! /bin/bash


export TVM_HOME=/root/work/tvm-ansor
export PYTHONPATH=$TVM_HOME/python
export TVM_LIBRARY_PATH=/root/work/tvm-ansor/build-release


# tasks_index=(1490)

# for idx in "${tasks_index[@]}"; do
#     python measure_programs.py --task-index "$idx"
# done


inputs=()
for f in /root/work/tvm-ansor/gallery/constrained_gen/data/to_measure_family_ansor/*.json; do
  inputs+=(--input "$f")
done

python /root/work/tvm-ansor/gallery/constrained_gen/data/measure_programs.py \
    "${inputs[@]}" \
    --output-dir /root/work/tvm-ansor/gallery/constrained_gen/data/measured_family_ansor