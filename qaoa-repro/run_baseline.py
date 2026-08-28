#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch runner for the Neural-QAOA-Squared MaxCut baseline reproduction.

Runs the official `QAOA_in_QAOA.py` (QAOA^2 / Neural-QAOA^2) across a small
grid of MaxCut instances, partition policies, and QAOA depths, then aggregates
the per-run result JSON files into a single CSV.

Official code limitation (verified experimentally): the `JointGenerator+Critic`
policy is only valid for depth == 1 because the pretrained joint generator emits
parameters with shape fixed to config.QAOA_DEPTH == 1, and QAOA.py asserts
len(init_gammas) == n_layers. All classical policies run at depth 1..3.

Usage:  E:/software/miniforge/python.exe run_baseline.py [--out results/baseline_results.csv]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import glob
from pathlib import Path

REPO = Path(__file__).resolve().parent / "Neural-QAOA-Squared"
SCRIPT = REPO / "competitors" / "QAOA-in-QAOA" / "QAOA_in_QAOA.py"
OSV = REPO / "data" / "instances" / "data" / "osv.json"
PY = r"E:/software/miniforge/python.exe"

INSTANCES = [
    ("bqp50-1",  "data/instances/data/test_instances_only/mc/bqp50-1.txt"),
    ("bqp50-2",  "data/instances/data/test_instances_only/mc/bqp50-2.txt"),
    ("be100.1",  "data/instances/data/test_instances_only/mc/be100.1.txt"),
    ("be100.2",  "data/instances/data/test_instances_only/mc/be100.2.txt"),
]

POLICIES = ["random", "modularity", "kl", "boundary", "JointGenerator+Critic"]
DEPTHS = {"random": [1, 2, 3], "modularity": [1, 2, 3], "kl": [1, 2, 3],
          "boundary": [1, 2, 3], "JointGenerator+Critic": [1]}
SUB_SIZE = 10
RUNS = 3


def greedy_maxcut(edges, n, restarts=20):
    """Simple classical greedy MaxCut reference (our own baseline, not from the paper)."""
    import random as rnd
    import numpy as np

    best = -1.0
    for _ in range(restarts):
        side = [0] * n
        order = list(range(n))
        rnd.shuffle(order)
        for v in order:
            cut_if_0 = sum(w for u, wv, w in edges if u == v and side[wv] == 1)
            cut_if_1 = sum(w for u, wv, w in edges if u == v and side[wv] == 0)
            side[v] = 0 if cut_if_0 >= cut_if_1 else 1
        val = sum(w for u, wv, w in edges if side[u] != side[wv])
        best = max(best, val)
    return best


def load_instance_edges(path):
    with open(path) as f:
        lines = f.read().strip().splitlines()
    n_v, n_e = map(int, lines[0].split())
    edges = []
    for line in lines[1:]:
        if line.strip():
            u, v, w = line.split()
            edges.append((int(u) - 1, int(v) - 1, float(w)))
    return n_v, edges


def latest_result_json(depth, policy, instance_name):
    """Return the most recent result JSON produced by the official script."""
    pattern = REPO / "result" / "main" / "test_instances_only" / \
        f"QAOA-in-QAOA_d{depth}_s{SUB_SIZE}_{policy}" / instance_name / "*" / "*.json"
    files = glob.glob(str(pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def run_one(instance_name, data_path, policy, depth, opt_value):
    env = dict(os.environ, TQDM_DISABLE="1")
    cmd = [PY, str(SCRIPT), "--data_path", str(REPO / data_path),
           "--experiment", "m", "--runs", str(RUNS), "--depth", str(depth),
           "--sub_size", str(SUB_SIZE), "--policy", policy, "--base", "qaoa",
           "--optimal_value", str(opt_value)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    wall = time.time() - t0
    if r.returncode != 0:
        return {"instance": instance_name, "policy": policy, "depth": depth,
                "ok": False, "error": (r.stderr or r.stdout)[-500:]}
    jf = latest_result_json(depth, policy, instance_name)
    if not jf:
        return {"instance": instance_name, "policy": policy, "depth": depth,
                "ok": False, "error": "no result json found"}
    j = json.load(open(jf))
    return {
        "instance": instance_name,
        "policy": policy,
        "depth": depth,
        "ok": True,
        "best_cut": j.get("best_cut_value"),
        "avg_cut": j.get("average_cut"),
        "best_ratio": j.get("best_approximation_ratio"),
        "avg_ratio": j.get("average_approximation_ratio"),
        "std_ratio": j.get("std_approximation_ratio"),
        "opt": opt_value,
        "avg_time_s": j.get("average_time"),
        "wall_s": round(wall, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/baseline_results.csv")
    ap.add_argument("--max-configs", type=int, default=0, help="limit configs (for smoke testing)")
    args = ap.parse_args()

    osv = json.load(open(OSV))
    results = []
    n_config = 0

    for instance_name, data_path in INSTANCES:
        opt = osv.get(instance_name)
        n_v, edges = load_instance_edges(REPO / data_path)

        # Our classical greedy reference (per instance)
        gval = greedy_maxcut(edges, n_v)
        results.append({"instance": instance_name, "policy": "greedy(classical)",
                        "depth": 0, "ok": True, "best_cut": gval,
                        "avg_cut": gval, "best_ratio": gval / opt if opt else None,
                        "avg_ratio": gval / opt if opt else None,
                        "std_ratio": 0.0, "opt": opt, "avg_time_s": "~0ms",
                        "wall_s": 0.0})

        for policy in POLICIES:
            for depth in DEPTHS[policy]:
                n_config += 1
                if args.max_configs and n_config > args.max_configs:
                    break
                print(f"[{time.strftime('%H:%M:%S')}] {instance_name} {policy} d{depth} ...", flush=True)
                res = run_one(instance_name, data_path, policy, depth, opt)
                res.setdefault("error", "")
                results.append(res)
                tag = "OK " if res["ok"] else "FAIL"
                detail = f"ratio={res.get('best_ratio')}" if res["ok"] else res.get("error", "")[-120:]
                print(f"  -> {tag} {detail}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "instance", "policy", "depth", "ok", "best_cut", "avg_cut",
            "best_ratio", "avg_ratio", "std_ratio", "opt", "avg_time_s", "wall_s", "error"])
        w.writeheader()
        for r in results:
            w.writerow(r)

    ok = sum(1 for r in results if r["ok"])
    print(f"\nDone. {ok}/{len(results)} configs OK. Results -> {out_path}")


if __name__ == "__main__":
    main()
