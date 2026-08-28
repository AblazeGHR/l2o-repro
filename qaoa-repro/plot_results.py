#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot approximation ratio (best-of-runs) vs QAOA depth per partition policy."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = Path(__file__).resolve().parent / "results" / "baseline_results.csv"
OUTDIR = Path(__file__).resolve().parent / "results"


def load_rows():
    with open(CSV) as f:
        return [r for r in csv.DictReader(f) if r["ok"] == "True"]


def main():
    rows = load_rows()
    instances = sorted({r["instance"] for r in rows})
    policies = ["random", "modularity", "kl", "boundary", "JointGenerator+Critic"]
    label = {"JointGenerator+Critic": "JointGen+Critic", "random": "random",
             "modularity": "modularity", "kl": "KL", "boundary": "boundary"}

    # Fig 1: per-instance, best ratio vs depth
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, inst in zip(axes.flat, instances):
        inst_rows = [r for r in rows if r["instance"] == inst]
        for p in policies:
            pr = sorted([r for r in inst_rows if r["policy"] == p], key=lambda r: int(r["depth"]))
            if not pr:
                continue
            xs = [int(r["depth"]) for r in pr]
            ys = [float(r["best_ratio"]) for r in pr]
            ax.plot(xs, ys, "o-", label=label[p])
        gr = [r for r in inst_rows if r["policy"].startswith("greedy")]
        if gr:
            ax.axhline(float(gr[0]["best_ratio"]), ls="--", color="gray", alpha=0.7, label="greedy (classical ref)")
        ax.set_title(inst)
        ax.set_xlabel("QAOA depth")
        ax.set_ylabel("Approx. ratio (best)")
        ax.set_xticks([0, 1, 2, 3])
        ax.grid(alpha=0.3)
    handles, labl = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labl, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Neural-QAOA$^2$ MaxCut baseline reproduction — approx. ratio vs QAOA depth", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    f1 = OUTDIR / "qaoa2_ratio_vs_depth_per_instance.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    # Fig 2: mean ratio across instances (only policies present at all depths 1..3)
    fig, ax = plt.subplots(figsize=(8, 5))
    for p in policies:
        pr = [r for r in rows if r["policy"] == p]
        if not pr:
            continue
        by_depth = {}
        for r in pr:
            d = int(r["depth"])
            by_depth.setdefault(d, []).append(float(r["best_ratio"]))
        depths = sorted(by_depth)
        means = [sum(by_depth[d]) / len(by_depth[d]) for d in depths]
        ax.plot(depths, means, "o-", label=label[p])
    ax.set_xlabel("QAOA depth")
    ax.set_ylabel("Mean approx. ratio over instances")
    ax.set_title("Neural-QAOA$^2$ MaxCut baseline — mean approx. ratio vs depth")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    f2 = OUTDIR / "qaoa2_mean_ratio_vs_depth.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)

    print(f"figures -> {f1}")
    print(f"figures -> {f2}")


if __name__ == "__main__":
    main()
