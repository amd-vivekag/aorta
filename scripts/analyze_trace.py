#!/usr/bin/env python3
"""
Analyze PyTorch profiler trace (Chrome trace format).
"""
import json
from collections import defaultdict
import re
import sys

TRACE_PATH = "/mnt/vast/huzhao/projects/aorta/data/shampoo_debug_trace.json"

# GPU categories (kernel, memcpy, etc.)
GPU_CATS = frozenset(("kernel", "gpu_memcpy", "gpu_memset", "ac2g", "gpu_user_annotation"))

# CPU categories
CPU_CAT_PATTERNS = ("cpu", "python", "user_annotation", "cuda_runtime", "fwdbwd")

# Collective patterns (case-insensitive)
COLLECTIVE_PATTERNS = [
    "alltoall", "all_to_all", "reduce_scatter", "allreduce", "all_reduce", "broadcast"
]

# Shampoo patterns
SHAMPOO_PATTERNS = ["shampoo", "precondition"]

# Step boundary patterns
STEP_PATTERNS = ["profilerstep", "iteration"]


def main():
    print("=" * 80)
    print("PYTORCH PROFILER TRACE ANALYSIS")
    print("\n=" * 80)
    print("\nLoading trace...")
    with open(TRACE_PATH, "r") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    print(f"Loaded {len(events)} events")

    # 1. Unique (pid, tid) pairs
    print("\n\n" + "=" * 80)
    print("1. UNIQUE (pid, tid) PAIRS AND STREAM/THREAD MAPPING")
    print("\n=" * 80)
    pid_tid_info = defaultdict(lambda: {"count": 0, "sample_names": [], "sample_cats": []})
    for e in events:
        pid = e.get("pid", -1)
        tid = e.get("tid", -1)
        key = (pid, tid)
        pid_tid_info[key]["count"] += 1
        if len(pid_tid_info[key]["sample_names"]) < 5:
            pid_tid_info[key]["sample_names"].append(e.get("name", ""))
        if len(pid_tid_info[key]["sample_cats"]) < 5:
            pid_tid_info[key]["sample_cats"].append(e.get("cat", ""))

    pairs = sorted(pid_tid_info.items(), key=lambda x: -x[1]["count"])
    for (pid, tid), info in pairs:
        names = info["sample_names"][:3]
        cats = list(set(info["sample_cats"]))[:3]
        print(f"  pid={pid}, tid={tid}: count={info['count']}, sample_names={names}, cats={cats}")
    print("\nTotal unique (pid, tid) pairs:", len(pairs))

    # 2. GPU events by stream
    print("\n\n" + "=" * 80)
    print("2. GPU EVENTS BY STREAM/TID")
    print("\n=" * 80)
    gpu_by_tid = defaultdict(lambda: {"count": 0, "ts_min": float("inf"), "ts_max": float("-inf"), "kernel_names": defaultdict(int)})
    for e in events:
        cat = e.get("cat", "")
        if cat not in GPU_CATS:
            continue
        pid = e.get("pid", -1)
        tid = e.get("tid", -1)
        key = (pid, tid)
        gpu_by_tid[key]["count"] += 1
        ts = e.get("ts", 0)
        if ts:
            gpu_by_tid[key]["ts_min"] = min(gpu_by_tid[key]["ts_min"], ts)
            gpu_by_tid[key]["ts_max"] = max(gpu_by_tid[key]["ts_max"], ts)
        name = e.get("name", "")
        gpu_by_tid[key]["kernel_names"][name] += 1

    for (pid, tid), info in sorted(gpu_by_tid.items(), key=lambda x: -x[1]["count"]):
        ts_min = info["ts_min"] if info["ts_min"] != float("inf") else 0
        ts_max = info["ts_max"] if info["ts_max"] != float("-inf") else 0
        print(f"\n  Stream (pid={pid}, tid={tid}):")
        print(f"    Total count: {info['count']}")
        print(f"    Time range: min_ts={ts_min:.2f}, max_ts={ts_max:.2f}")
        top10 = sorted(info["kernel_names"].items(), key=lambda x: -x[1])[:10]
        print("    Top 10 kernel names:")
        for name, cnt in top10:
            short = name[:80] + "..." if len(name) > 80 else name
            print(f"      {cnt:6d}x  {short}")

    # 3. CPU events
    print("\n\n" + "=" * 80)
    print("3. CPU EVENTS - TOP 20 OPERATIONS")
    print("\n=" * 80)
    cpu_ops = defaultdict(lambda: {"count": 0, "durations": []})
    for e in events:
        cat = e.get("cat", "").lower()
        if not any(p in cat for p in CPU_CAT_PATTERNS):
            continue
        name = e.get("name", "")
        dur = e.get("dur", 0) or 0
        cpu_ops[name]["count"] += 1
        if dur:
            cpu_ops[name]["durations"].append(dur)

    cpu_sorted = sorted(cpu_ops.items(), key=lambda x: -x[1]["count"])[:20]
    for name, info in cpu_sorted:
        avg_dur = sum(info["durations"]) / len(info["durations"]) if info["durations"] else 0
        short = name[:70] + "..." if len(name) > 70 else name
        print(f"  {info['count']:7d}x  avg_dur={avg_dur:10.2f}us  {short}")

    # 4. Shampoo events
    print("\n\n" + "=" * 80)
    print("4. SHAMPOO / PRECONDITION EVENTS")
    print("\n=" * 80)
    shampoo_events = [e for e in events if any(p in e.get("name", "").lower() for p in ["shampoo", "precondition"])]
    if not shampoo_events:
        print("  No events found.")
    else:
        for i, e in enumerate(shampoo_events[:100]):
        print(f"\n  Event {i+1}: name={e.get('name','')}")
            print(f"    stream: pid={e.get('pid')}, tid={e.get('tid')}")
            print(f"    ts={e.get('ts',0):.2f}, dur={e.get('dur',0):.2f}us")
            print(f"    args: {str(e.get('args',{}))[:200]}")
        if len(shampoo_events) > 100:
            print(f"
  ... and {len(shampoo_events)-100} more")

    # 5. Collective events
    print("\n\n" + "=" * 80)
    print("5. NCCL/RCCL COLLECTIVE EVENTS")
    print("\n=" * 80)
    collective_events = []
    collective_by_type = defaultdict(list)
    for e in events:
        name = e.get("name", "").lower()
        if any(p in name for p in COLLECTIVE_PATTERNS):
            collective_events.append(e)
            for p in COLLECTIVE_PATTERNS:
                if p in name:
                    collective_by_type[p].append(e)
                    break

    print(f"
Total: {len(collective_events)}")
    for p in COLLECTIVE_PATTERNS:
        cnt = len(collective_by_type[p])
        if cnt:
            print(f"  {p}: {cnt}")
    print("\nSample (first 50):")
    for i, e in enumerate(collective_events[:50]):
        print(f"  {i+1}. {e.get('name','')} | pid={e.get('pid')} tid={e.get('tid')} ts={e.get('ts',0):.0f} dur={e.get('dur',0):.2f}us")

    # 6. Backward vs Optimizer
    print("\n\n" + "=" * 80)
    print("6. BACKWARD/AUTOGRAD vs OPTIMIZER/STEP")
    print("\n=" * 80)
    backward_events = [e for e in events if "backward" in e.get("name", "").lower() or "autograd" in e.get("name", "").lower()]
    optimizer_events = [e for e in events if "optimizer" in e.get("name", "").lower() or "step" in e.get("name", "").lower()]

    print(f"
Backward/Autograd: {len(backward_events)}")
    bwd_names = defaultdict(int)
    for e in backward_events:
        bwd_names[e.get("name", "")] += 1
    for name, cnt in sorted(bwd_names.items(), key=lambda x: -x[1])[:15]:
        print(f"  {cnt:6d}x  {name[:70]}")

    print(f"
Optimizer/Step: {len(optimizer_events)}")
    opt_names = defaultdict(int)
    for e in optimizer_events:
        opt_names[e.get("name", "")] += 1
    for name, cnt in sorted(opt_names.items(), key=lambda x: -x[1])[:15]:
        print(f"  {cnt:6d}x  {name[:70]}")

    # 7. Time range and steps
    print("\n\n" + "=" * 80)
    print("7. OVERALL TIME RANGE AND STEP BOUNDARIES")
    print("\n=" * 80)
    all_ts = [e.get("ts", 0) for e in events if e.get("ts")]
    if all_ts:
        ts_min, ts_max = min(all_ts), max(all_ts)
        print(f"
Time range: ts_min={ts_min:.2f}, ts_max={ts_max:.2f}")
        print(f"Total span: {(ts_max - ts_min) / 1e6:.2f} seconds")

    profiler_steps = [e for e in events if "profilerstep" in e.get("name", "").lower()]
    print(f"
ProfilerStep events: {len(profiler_steps)}")
    if profiler_steps:
        steps_sorted = sorted(profiler_steps, key=lambda e: e.get("ts", 0))
        print("  First 10:")
        for e in steps_sorted[:10]:
            print(f"    {e.get('name','')} ts={e.get('ts',0):.0f} dur={e.get('dur',0):.2f}us")
        print("  Last 5:")
        for e in steps_sorted[-5:]:
            print(f"    {e.get('name','')} ts={e.get('ts',0):.0f} dur={e.get('dur',0):.2f}us")

if __name__ == "__main__":
    main()
