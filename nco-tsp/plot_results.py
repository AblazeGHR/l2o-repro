#!/usr/bin/env python
"""Parse a run.py training log and plot convergence curves.

Usage:
    python plot_results.py <train.log> [out.png]

Expected log lines (from run.py / train.py):
    epoch: {epoch}, train_batch_id: {batch_id}, avg_cost: {cost}       # train (sampling)
    Validation overall avg_cost: {cost} +- {err}                        # val (greedy)
"""
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_RE = re.compile(r"epoch: (\d+), train_batch_id: (\d+), avg_cost: ([\d.eE+-]+)")
VAL_RE = re.compile(r"Validation overall avg_cost: ([\d.eE+-]+) \+-\s*([\d.eE+-]+)")


def main():
    log_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "training_curves.png"

    train_steps, train_costs = [], []
    val_epochs, val_costs = [], []

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = TRAIN_RE.search(line)
            if m:
                epoch = int(m.group(1))
                batch = int(m.group(2))
                # global step = epoch * batches_per_epoch + batch; approximate via epoch only
                step = epoch + batch / 1000.0
                train_steps.append(step)
                train_costs.append(float(m.group(3)))
                continue
            m = VAL_RE.search(line)
            if m:
                val_epochs.append(len(val_costs))
                val_costs.append(float(m.group(1)))

    if not train_costs and not val_costs:
        print(f"No metrics found in {log_path}")
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(train_steps, train_costs, lw=0.7, alpha=0.75, label="train avg_cost (sampling)")
    if val_costs:
        ax.plot(val_epochs, val_costs, lw=2, marker="o", ms=3, label="val avg_cost (greedy)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("tour length")
    ax.set_title("Training convergence")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    if val_costs:
        ax.plot(val_epochs, val_costs, lw=2, marker="o", ms=4, color="C1")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val tour length (greedy)")
        ax.set_title("Validation curve")
        ax.grid(alpha=0.3)
        best = min(val_costs)
        ax.axhline(best, ls="--", color="gray", alpha=0.7, label=f"best = {best:.3f}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no validation data", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    print(f"train samples: {len(train_costs)}, val samples: {len(val_costs)}")
    if val_costs:
        print(f"val: first={val_costs[0]:.3f}, last={val_costs[-1]:.3f}, best={min(val_costs):.3f}")


if __name__ == "__main__":
    main()
