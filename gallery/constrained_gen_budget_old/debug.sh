python debug_latent_decode.py \
  --checkpoint /root/work/tvm-ansor/gallery/constrained_gen_budget/checkpoints/last.pt \
  --record-json '/root/work/tvm-ansor/gallery/constrained_gen/data/measured_ansor/1490_([3eda1939e30b947e921f5e1814346365,[1,56,56,128],[6,6,32,128],[1,56,56,32]],cuda).json' \
  --network-info-folder /root/work/tvm-ansor/gallery/dataset/network_info_all \
  --device cuda \
  --latent-gradient \
  --num-steps 10 \
  --step-size 0.2 \
  --output debug_report.json
  # --output debug_report_latent.json