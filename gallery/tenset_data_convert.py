"""Convert tenset-era measure_records (*.json) and network_info (*.task.pkl)
so that they can be loaded by the current tvm-ansor build.

Changes applied:
  - Target string: `shared_memory_per_block=` -> `max_shared_memory_per_block=`
  - Target JSON (inside pickled SearchTask):
      * Map key `shared_memory_per_block` -> `max_shared_memory_per_block`
      * TargetKind attr `device_type` -> `default_device_type`
      * Inject required `features` field into each Target node (empty Map)
  - workload_key: flat shape encoding -> nested shape encoding
      (hash stays the same; shape boundaries are recovered from compute_dag.tensors)

Result values `r` in JSON record files are left untouched.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys


def transform_target_json(js: str) -> str:
    """Transform a TVM-serialized JSON blob so its Target/TargetKind/Map nodes
    match the current tvm-ansor schema. Safe to call on any JSON blob — only
    the matching nodes are rewritten."""
    obj = json.loads(js)
    nodes = obj.get("nodes")
    if not isinstance(nodes, list):
        return js

    for n in nodes:
        if not isinstance(n, dict):
            continue
        tk = n.get("type_key")
        if tk == "Map":
            keys = n.get("keys")
            if isinstance(keys, list):
                n["keys"] = [
                    "max_shared_memory_per_block" if k == "shared_memory_per_block" else k
                    for k in keys
                ]
        elif tk == "TargetKind":
            attrs = n.get("attrs")
            if isinstance(attrs, dict) and "device_type" in attrs and "default_device_type" not in attrs:
                attrs["default_device_type"] = attrs.pop("device_type")

    # Inject `features` into every Target node. Append one empty Map node
    # per Target at the end of the nodes array and reference it by index.
    for n in nodes:
        if isinstance(n, dict) and n.get("type_key") == "Target":
            attrs = n.get("attrs")
            if isinstance(attrs, dict) and "features" not in attrs:
                attrs["features"] = str(len(nodes))
                nodes.append({"type_key": "Map"})

    return json.dumps(obj, indent=2)


def install_load_hook():
    """Monkey-patch __setstate__ so pickled tenset tasks load on current tvm-ansor:
      * Object.__setstate__: rewrite handle JSON for Target schema changes.
      * SearchTask.__setstate__: backfill `desc` field added after tenset era.
    """
    from tvm.runtime.object import Object
    from tvm.auto_scheduler.search_task import SearchTask

    obj_orig = Object.__setstate__

    def obj_patched(self, state):
        handle = state.get("handle") if isinstance(state, dict) else None
        if isinstance(handle, str):
            try:
                state = {"handle": transform_target_json(handle)}
            except Exception:
                pass
        obj_orig(self, state)

    Object.__setstate__ = obj_patched

    task_orig = SearchTask.__setstate__

    def task_patched(self, state):
        if isinstance(state, dict) and "desc" not in state:
            state = dict(state)
            state["desc"] = ""
        task_orig(self, state)

    SearchTask.__setstate__ = task_patched


def _rebuild_task_with_new_key(t):
    """Return a SearchTask with workload_key in nested-shape form derived from
    ``t.compute_dag``. The hash component is preserved."""
    from tvm.auto_scheduler import SearchTask

    new_key = t.compute_dag.workload_key()
    if new_key == t.workload_key:
        return t, new_key
    return (
        SearchTask(
            compute_dag=t.compute_dag,
            workload_key=new_key,
            target=t.target,
            target_host=t.target.host,
            hardware_params=t.hardware_params,
            layout_rewrite_option=t.layout_rewrite_option,
            task_inputs=list(t.task_input_names),
            desc=t.desc,
        ),
        new_key,
    )


def convert_pkl(src_path: str, dst_path: str, wk_map: dict | None = None) -> None:
    """Load a pickle of SearchTasks, rebuild each task so its workload_key is
    in the nested-shape form, and write the result back.

    Also accumulates the old->new workload_key mapping into ``wk_map`` so
    the JSON-record pass can rewrite matching record files.

    Supports the two pickle shapes found in tenset data:
      * ``(list[SearchTask], list[int])`` — (tasks, weights) for *.task.pkl
      * ``list[SearchTask]``              — for all_tasks.pkl
    """
    with open(src_path, "rb") as f:
        obj = pickle.load(f)

    def _rebuild_list(tasks):
        new = []
        for t in tasks:
            rebuilt, new_key = _rebuild_task_with_new_key(t)
            if wk_map is not None:
                wk_map.setdefault(t.workload_key, new_key)
            new.append(rebuilt)
        return new

    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], list):
        obj = (_rebuild_list(obj[0]), obj[1])
    elif isinstance(obj, list):
        obj = _rebuild_list(obj)
    # else: unknown shape — leave as is (e.g. relay.pkl shouldn't reach here)

    tmp = dst_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, dst_path)


def convert_json_workload_key(path: str, wk_map: dict) -> bool:
    """Rewrite the workload_key string appearing on every record line.

    All records in a single file share one workload_key, so a one-shot binary
    replace suffices. Returns True if the file was modified.
    """
    with open(path, "rb") as f:
        data = f.read()
    # Read just the first non-empty line to discover this file's workload_key.
    nl = data.find(b"\n")
    head = data[:nl] if nl >= 0 else data
    try:
        rec = json.loads(head.decode("utf-8"))
        old_key = rec["i"][0][0]
    except (ValueError, IndexError, KeyError, UnicodeDecodeError):
        return False
    new_key = wk_map.get(old_key)
    if new_key is None or new_key == old_key:
        return False

    # The workload_key is stored as a JSON string value, so its on-disk form is
    # the JSON-escaped version (including surrounding quotes). Compute both and
    # do a single binary substitution.
    old_token = json.dumps(old_key).encode("utf-8")
    new_token = json.dumps(new_key).encode("utf-8")
    if old_token not in data:
        return False
    new_data = data.replace(old_token, new_token)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(new_data)
    os.replace(tmp, path)
    return True


def convert_json_records(src_path: str, dst_path: str) -> None:
    """Rewrite the one target-attr rename in each record line.

    The only attribute that appears literally as ``shared_memory_per_block=`` in
    measure_records JSON is the target string field; no other field uses that
    exact token. Pure string replacement is safe and ~100x faster than
    re-parsing every record.
    """
    with open(src_path, "rb") as f:
        src = f.read()
    needle = b"shared_memory_per_block="
    # Avoid re-touching already-converted files.
    if needle not in src or b"max_shared_memory_per_block=" in src:
        if needle not in src:
            return
    # Replace only occurrences not already prefixed by `max_`.
    new = src.replace(b"-shared_memory_per_block=", b"-max_shared_memory_per_block=")
    if new == src:
        return
    tmp = dst_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(new)
    os.replace(tmp, dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records-dir",
        default="/root/work/tvm-ansor/gallery/dataset/measure_records_tenset",
        help="Directory containing tenset JSON measure records (recursive).",
    )
    parser.add_argument(
        "--network-info-dir",
        default="/root/work/tvm-ansor/gallery/dataset/network_info_tenset",
        help="Directory containing tenset .task.pkl files.",
    )
    parser.add_argument(
        "--skip-target",
        action="store_true",
        help="Skip target/schema conversion (use if previous run already did it).",
    )
    parser.add_argument(
        "--skip-workload-key",
        action="store_true",
        help="Skip flat->nested workload_key conversion.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    install_load_hook()

    json_files = sorted(glob.glob(os.path.join(args.records_dir, "**", "*.json"), recursive=True))
    task_pkls = sorted(glob.glob(os.path.join(args.network_info_dir, "*.task.pkl")))
    all_tasks_pkl = os.path.join(args.network_info_dir, "all_tasks.pkl")
    aggregate_pkls = [all_tasks_pkl] if os.path.exists(all_tasks_pkl) else []
    all_pkls = task_pkls + aggregate_pkls

    print(f"JSON records:   {len(json_files)} files")
    print(f"Task pkls:      {len(task_pkls)} files")
    print(f"Aggregate pkls: {len(aggregate_pkls)} files")

    if args.dry_run:
        print("(dry-run) not writing anything")
        return

    # Phase 1: target-string rewrite in JSON record files (cheap, per-file).
    if not args.skip_target:
        for i, p in enumerate(json_files):
            convert_json_records(p, p)
            if (i + 1) % 1000 == 0 or i + 1 == len(json_files):
                print(f"  [json/target] {i+1}/{len(json_files)}", flush=True)

    # Phase 2: rebuild .task.pkl / all_tasks.pkl with nested workload_key,
    # collecting an old->new mapping as we go.
    wk_map: dict = {}
    if not args.skip_workload_key:
        for i, p in enumerate(all_pkls):
            convert_pkl(p, p, wk_map=wk_map)
            if (i + 1) % 25 == 0 or i + 1 == len(all_pkls):
                print(f"  [pkl]  {i+1}/{len(all_pkls)}  (mapping size={len(wk_map)})", flush=True)
    else:
        # Still need to (re)build the mapping so JSON records can be rewritten.
        for p in all_pkls:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            tasks = obj[0] if (isinstance(obj, tuple) and isinstance(obj[0], list)) else obj
            if not isinstance(tasks, list):
                continue
            for t in tasks:
                wk_map.setdefault(t.workload_key, t.compute_dag.workload_key())

    # Phase 3: rewrite workload_key in JSON record files using the mapping.
    if not args.skip_workload_key:
        rewritten = 0
        for i, p in enumerate(json_files):
            if convert_json_workload_key(p, wk_map):
                rewritten += 1
            if (i + 1) % 1000 == 0 or i + 1 == len(json_files):
                print(f"  [json/wk]  {i+1}/{len(json_files)}  rewritten={rewritten}", flush=True)

    print("done")


if __name__ == "__main__":
    main()
