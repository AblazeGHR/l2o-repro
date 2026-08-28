# -*- coding: utf-8 -*-
"""结果分析：收敛对比图 / 汇总 CSV / 每实例最佳超参 / LLM 理由抽样 / 汇总指标。

依赖 results/checkpoints/{llm_hpo,tpe,random}_inst_XX.jsonl。
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
CKPT = os.path.join(RESULTS, "checkpoints")

METHODS = ["llm_hpo", "tpe", "random"]
METHOD_LABEL = {"llm_hpo": "LLM-HPO", "tpe": "Optuna TPE", "random": "Random Search"}


def load_all():
    """返回 {method: {inst: {"costs": [float...], "best": {...}}}}"""
    data = {}
    insts = sorted({int(f.split("_")[-1].split(".")[0]) for f in os.listdir(CKPT)}) or [0]
    for m in METHODS:
        data[m] = {}
        for i in insts:
            p = os.path.join(CKPT, f"{m}_inst_{i:02d}.jsonl")
            if not os.path.exists(p):
                continue
            costs, best_h = [], None
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if m == "llm_hpo":
                        for s in row["suggestions"]:
                            costs.append(s["best_cost"])
                            if best_h is None or s["best_cost"] < best_h["best_cost"]:
                                best_h = dict(s)
                    else:
                        costs.append(row["cost"])
                        if best_h is None or row["cost"] < best_h["cost"]:
                            best_h = {"h": row["h"], "cost": row["cost"]}
            data[m][i] = {"costs": costs, "best": best_h}
    return data


def running_best(costs):
    out, b = [], float("inf")
    for c in costs:
        b = min(b, c)
        out.append(b)
    return np.asarray(out, dtype=float)


def plot_convergence(data, out_png):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"llm_hpo": "#d62728", "tpe": "#1f77b4", "random": "#7f7f7f"}
    min_len = min(len(data[m][i]["costs"]) for m in METHODS for i in data[m])
    x = np.arange(1, min_len + 1)
    for m in METHODS:
        mat = np.stack([running_best(data[m][i]["costs"])[:min_len] for i in sorted(data[m])])
        mean, std = mat.mean(axis=0), mat.std(axis=0)
        ax.plot(x, mean, label=METHOD_LABEL[m], color=colors[m], lw=2)
        ax.fill_between(x, mean - std, mean + std, color=colors[m], alpha=0.15)
    ax.set_xlabel("评估次数（GA 超参尝试次数）")
    ax.set_ylabel("最优路径成本（越小越好，均值 ± 标准差）")
    ax.set_title(f"LLM-HPO vs Optuna TPE vs Random Search 收敛对比（{len(data['llm_hpo'])} 个 TSP20 实例）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("saved", out_png)


def main():
    data = load_all()
    if not data["llm_hpo"]:
        print("无数据，请先运行 run_experiment.py")
        sys.exit(1)

    plot_convergence(data, os.path.join(RESULTS, "convergence.png"))

    # 汇总 CSV：实例 / 方法 / 最终成本 / 最佳超参 / LLM 调用次数 / tokens
    rows_csv = []
    insts = sorted(data["llm_hpo"].keys())
    for i in insts:
        for m in METHODS:
            d = data[m].get(i)
            if d is None:
                continue
            costs = d["costs"]
            final_cost = min(costs)
            best = d["best"]
            if m == "llm_hpo":
                bh = best
                row = {
                    "instance": i, "method": METHOD_LABEL[m], "final_cost": f"{final_cost:.4f}",
                    "best_population": bh["population"], "best_crossover_rate": bh["crossover_rate"],
                    "best_mutation_rate": bh["mutation_rate"], "best_generations": bh["generations"],
                    "n_evaluations": len(costs), "llm_calls": len(costs) // 8,
                    "prompt_tokens": "", "completion_tokens": "", "total_tokens": "",
                }
            else:
                bh = best
                row = {
                    "instance": i, "method": METHOD_LABEL[m], "final_cost": f"{final_cost:.4f}",
                    "best_population": bh["h"]["population"], "best_crossover_rate": bh["h"]["crossover_rate"],
                    "best_mutation_rate": bh["h"]["mutation_rate"], "best_generations": bh["h"]["generations"],
                    "n_evaluations": len(costs), "llm_calls": 0,
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                }
            rows_csv.append(row)

    # tokens：从 llm_hpo checkpoint 累计
    tot_p, tot_c, tot_calls = 0, 0, 0
    for i in insts:
        p = os.path.join(CKPT, f"llm_hpo_inst_{i:02d}.jsonl")
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                tot_calls += 1
                u = row.get("usage") or {}
                tot_p += u.get("prompt_tokens", 0)
                tot_c += u.get("completion_tokens", 0)
    for r in rows_csv:
        if r["method"] == "LLM-HPO":
            r["llm_calls"] = tot_calls
            r["prompt_tokens"] = tot_p
            r["completion_tokens"] = tot_c
            r["total_tokens"] = tot_p + tot_c

    import csv
    with open(os.path.join(RESULTS, "summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    print("saved", os.path.join(RESULTS, "summary.csv"))

    # LLM 理由抽样存档
    sample_insts = [0, 7, 14, 21, 29]
    sample_rounds = [0, 5, 10, 15, 20, 25, 29]
    out = []
    for i in sample_insts:
        p = os.path.join(CKPT, f"llm_hpo_inst_{i:02d}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row["round"] in sample_rounds:
                    out.append({
                        "instance": i, "round": row["round"],
                        "suggestions": [{"population": s["population"], "crossover_rate": s["crossover_rate"],
                                         "mutation_rate": s["mutation_rate"], "generations": s["generations"],
                                         "reason": s.get("reason", ""), "best_cost": s["best_cost"]}
                                        for s in row["suggestions"]],
                    })
    with open(os.path.join(RESULTS, "llm_reasons_sample.jsonl"), "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("saved", os.path.join(RESULTS, "llm_reasons_sample.jsonl"))

    # 汇总指标
    agg = {}
    for m in METHODS:
        finals = [min(data[m][i]["costs"]) for i in data[m]]
        agg[m] = {"mean_final": float(np.mean(finals)), "median_final": float(np.median(finals)),
                  "std_final": float(np.std(finals)), "min_final": float(np.min(finals))}
    llm, tpe, rnd = agg["llm_hpo"], agg["tpe"], agg["random"]
    agg["improve_vs_tpe"] = (tpe["mean_final"] - llm["mean_final"]) / tpe["mean_final"] * 100
    agg["improve_vs_random"] = (rnd["mean_final"] - llm["mean_final"]) / rnd["mean_final"] * 100
    agg["total_llm_calls"] = tot_calls
    agg["prompt_tokens"] = tot_p
    agg["completion_tokens"] = tot_c
    agg["total_tokens"] = tot_p + tot_c
    agg["cost_yuan_input"] = tot_p / 1e6 * 0.8
    agg["cost_yuan_output"] = tot_c / 1e6 * 2
    agg["cost_yuan_total"] = agg["cost_yuan_input"] + agg["cost_yuan_output"]
    with open(os.path.join(RESULTS, "aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
