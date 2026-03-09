"""
dump_sketches.py — to_measure_programs/ 내 모든 JSON 파일을 스캔하여
                   (workload_key, sketch_fingerprint) 기준 고유 sketch 대표 record를
                   all_sketches.json 파일에 모아 저장.

사용법:
    python /root/work/tvm-ansor/gallery/constrained_gen/dump_sketches.py
"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(__file__))

from modules.record_loader import state_sketch_fingerprint
from tvm.auto_scheduler.measure_record import load_records, dump_record_to_string

# ─── 경로 ───
PROGRAMS_DIR = "/root/work/tvm-ansor/gallery/dataset/to_measure_programs"
OUTPUT_PATH = os.path.join(PROGRAMS_DIR, "all_sketches.json")

# ─── 모든 JSON 파일 (all_sketches.json 제외) ───
json_files = sorted(
    f for f in glob.glob(os.path.join(PROGRAMS_DIR, "*.json"))
    if os.path.basename(f) != "all_sketches.json"
)

print(f"Scanning {len(json_files)} JSON files in {PROGRAMS_DIR}")

seen = set()   # (workload_key, sketch_fingerprint)
lines = []

for ji, json_path in enumerate(json_files):
    fname = os.path.basename(json_path)
    count = 0
    for inp, res in load_records(json_path):
        wkey = inp.task.workload_key
        fp = state_sketch_fingerprint(inp.state)
        key = (wkey, fp)
        if key not in seen:
            seen.add(key)
            lines.append(dump_record_to_string(inp, res))
            count += 1
    if count > 0:
        print(f"  [{ji+1}/{len(json_files)}] {fname}: +{count} new sketches")

# ─── 파일 저장 ───
with open(OUTPUT_PATH, "w") as f:
    for line in lines:
        f.write(line)

print(f"\n=== 완료: {len(json_files)} files scanned, "
      f"총 {len(lines)} unique sketches → {OUTPUT_PATH} ===")
