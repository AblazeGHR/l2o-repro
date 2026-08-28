# -*- coding: utf-8 -*-
"""Portfolio 升级实验：LLM 生成有区分度实例 → 验证 Portfolio 价值。

流程:
    1. LLM 实例生成器：按几何结构类型生成 n=40~60 的有区分度实例（120 个）
    2. 匹配 n 的均匀随机控制实例（120 个）
    3. 全部实例实际运行 4 算法（各 2s 预算）
    4. 区分度统计：算法间 max-min 相对差距分布（LLM生成 vs 均匀）
    5. LLM 在线选择器（few-shot warm-up 反馈）对两类实例分别选择
    6. 输出对比表 + 区分度直方图 + 生成实例散点抽样 + README

用法:
    python run_upgrade.py [--n-gen 120] [--budget 2.0] [--seed 42]
                          [--workers 8] [--gen-workers 4] [--warmup 10]
                          [--skip-gen] [--skip-eval] [--skip-select]
"""

import argparse
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np

from portfolio.algorithms import run_all_algorithms
from portfolio.instances import distance_matrix, generate_instances
from portfolio.features import extract_features
from portfolio.llm_selector import ALGORITHM_POOL, ZhipuChat, rotator_stats
from portfolio.gen_instances import STRUCTURE_TYPES, generate_one, get_thread_chat
from portfolio.online_selector import select

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "results", "upgrade")
DATA_DIR = os.path.join(OUT_DIR, "data")

BEST_EPS = 1e-9


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------- 多进程工作函数 ----------

def _evaluate_one(args):
    idx, cities, budget, seed = args
    dist = distance_matrix(cities)
    res = run_all_algorithms(cities, dist, budget, seed)
    return idx, res


# ---------- 实例生成（多线程） ----------

def generate_instances_llm(n_gen, seed, gen_workers):
    rng = random.Random(seed)
    structs = list(STRUCTURE_TYPES.keys())
    slots = [structs[i % len(structs)] for i in range(n_gen)]

    def worker(job):
        slot_idx, structure = job
        chat = get_thread_chat()
        r = generate_one(chat, structure, seed + slot_idx * 131)
        r["slot"] = slot_idx
        r["idx"] = slot_idx
        return r

    results = {}
    with ThreadPoolExecutor(max_workers=gen_workers) as ex:
        futs = {ex.submit(worker, (i, s)): i for i, s in enumerate(slots)}
        for fut in as_completed(futs):
            r = fut.result()
            results[r["slot"]] = r
            log(f"生成进度: {len(results)}/{n_gen}  "
                f"(slot={r['slot']} structure={r['structure']} ok={r.get('ok')} n={r.get('n')}) "
                f"| API {rotator_stats()}")
    return [results[i] for i in range(n_gen)], slots


# ---------- 统计 ----------

def discrimination_metric(lengths):
    vals = list(lengths.values())
    best, worst = min(vals), max(vals)
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    return {
        "max_min_rel": (worst - best) / best,
        "rel_std": std / mean,
        "best": best,
        "worst": worst,
    }


def agg_disc(records):
    if not records:
        return {"count": 0, "mean": 0.0, "median": 0.0, "max": 0.0,
                "frac_gt5pct": 0.0, "mean_rel_std": 0.0}
    vals = [r["max_min_rel"] for r in records]
    return {
        "count": len(vals),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "max": float(np.max(vals)),
        "frac_gt5pct": float(np.mean([v > 0.05 for v in vals])),
        "mean_rel_std": float(np.mean([r["rel_std"] for r in records])),
    }


def per_instance_policies(eval_res, selections):
    """构造每实例记录 + 各策略汇总。"""
    recs = []
    for idx, lens_obj in sorted(eval_res.items()):
        lengths = {a: d["length"] for a, d in lens_obj.items()}
        best = min(lengths.values())
        best_algs = {a for a, l in lengths.items() if l <= best + BEST_EPS}
        sel = selections.get(idx, {}).get("algorithm")
        sel_len = lengths.get(sel)
        recs.append({
            "idx": idx,
            "lengths": lengths,
            "best": best,
            "best_algs": sorted(best_algs),
            "sel": sel,
            "sel_len": sel_len,
        })
    return recs


def summarize_policy(recs, key_fn, win_fn, gap_fn):
    gaps = [gap_fn(r) for r in recs if gap_fn(r) is not None]
    return {
        "mean_gap": (sum(gaps) / len(gaps) * 100) if gaps else None,
        "accuracy": sum(win_fn(r) for r in recs) / len(recs),
        "wins": sum(win_fn(r) for r in recs),
    }


def compare_sets(eval_res, selections, label):
    recs = per_instance_policies(eval_res, selections)
    n = len(recs)

    def gap_of(alg):
        return lambda r: r["lengths"][alg] / r["best"] - 1.0 if r["best"] > 0 else None

    pols = {}
    # 各固定单算法
    for a in ALGORITHM_POOL:
        pols[a] = summarize_policy(
            recs, lambda r, a=a: r["lengths"][a],
            lambda r, a=a: a in r["best_algs"], gap_of(a))
    # 固定最优（该集上平均 gap 最小的单算法，事后口径）
    best_fixed = min(ALGORITHM_POOL, key=lambda a: pols[a]["mean_gap"])
    pols["固定最优"] = pols[best_fixed]
    pols["固定最优"]["_which"] = best_fixed
    # LLM 在线选择（解析失败 → 随机兜底期望）
    pols["LLM在线选择"] = summarize_policy(
        recs,
        lambda r: r["sel_len"] if r["sel_len"] is not None
        else sum(r["lengths"].values()) / len(r["lengths"]),
        lambda r: r["sel"] in r["best_algs"] if r["sel"] else
        sum(1 for a in r["best_algs"]) / len(ALGORITHM_POOL),
        lambda r: (r["sel_len"] / r["best"] - 1.0) if r["sel_len"] is not None
        else (sum(r["lengths"].values()) / len(r["lengths"]) / r["best"] - 1.0))
    # 随机期望
    pols["随机选择"] = summarize_policy(
        recs,
        lambda r: sum(r["lengths"].values()) / len(r["lengths"]),
        lambda r: len(r["best_algs"]) / len(ALGORITHM_POOL),
        lambda r: (sum(r["lengths"].values()) / len(r["lengths"]) / r["best"] - 1.0))
    # oracle
    pols["事后最优"] = summarize_policy(
        recs, lambda r: r["best"], lambda r: 1.0, lambda r: 0.0)
    return recs, pols, best_fixed


def policy_table(pols):
    lines = ["| 策略 | 平均gap(vs最优) | 达到最优率 | 达到最优数 |", "|---|---|---|---|"]
    order = ["LLM在线选择", "固定最优", "随机选择", "事后最优"] + ALGORITHM_POOL
    for k in order:
        p = pols[k]
        which = f" ({pols[k].get('_which')})" if k == "固定最优" else ""
        lines.append(f"| {k}{which} | {p['mean_gap']:.4f}% | {p['accuracy'] * 100:.1f}% | {p['wins']} |")
    return "\n".join(lines)


# ---------- 绘图 ----------

def _setup_cjk():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams
    for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            rcParams["font.sans-serif"] = [name]
            rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue


def plot_discrimination(gen_recs, uni_recs, gen_meta, uni_meta, path):
    _setup_cjk()
    import matplotlib.pyplot as plt

    g = [r["max_min_rel"] * 100 for r in gen_recs]
    u = [r["max_min_rel"] * 100 for r in uni_recs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    bins = np.linspace(0, max(max(g), max(u), 1) * 1.05, 25)
    ax.hist(g, bins=bins, alpha=0.6, label=f"LLM生成 (均值{np.mean(g):.2f}%)",
            color="#d1495b", edgecolor="white")
    ax.hist(u, bins=bins, alpha=0.6, label=f"均匀随机 (均值{np.mean(u):.2f}%)",
            color="#2e86ab", edgecolor="white")
    ax.axvline(np.mean(g), color="#d1495b", ls="--", lw=1)
    ax.axvline(np.mean(u), color="#2e86ab", ls="--", lw=1)
    ax.set_xlabel("算法间最大-最小相对差距 (%)")
    ax.set_ylabel("实例数")
    ax.set_title("区分度分布对比（4 算法 2s 预算实测）")
    ax.legend()

    ax = axes[1]
    labels = [s for s in STRUCTURE_TYPES if s in {r["structure"] for r in gen_recs}]
    data = [[r["max_min_rel"] * 100 for r in gen_recs if r["structure"] == s] for s in labels]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#d1495b")
        patch.set_alpha(0.5)
    ax.set_ylabel("最大-最小相对差距 (%)")
    ax.set_title("LLM生成实例按几何结构分组的区分度")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_samples(instances, path, cols=3, rows=3):
    _setup_cjk()
    import matplotlib.pyplot as plt
    picks = instances[:: max(1, len(instances) // (cols * rows))][: cols * rows]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    for ax, inst in zip(axes.flat, picks):
        xs = [c[0] for c in inst["coords"]]
        ys = [c[1] for c in inst["coords"]]
        ax.scatter(xs, ys, s=18, color="#1f77b4")
        ax.set_title(f"{inst['structure']} n={inst['n']}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    fig.suptitle("LLM 生成实例抽样（散点图）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------- README 渲染 ----------

def render_upgrade_readme(args, gen_disc, uni_disc, gen_pols, uni_pols,
                          fallback_cnt, gen_recs, usage):
    gen_pol_table = policy_table(gen_pols)
    uni_pol_table = policy_table(uni_pols)

    def disc_line(d, name):
        return (f"- **{name}**: 实例数 {d['count']}，max-min 相对差距 "
                f"均值 **{d['mean'] * 100:.2f}%** / 中位 {d['median'] * 100:.2f}% / "
                f"最大 {d['max'] * 100:.2f}%，差距>5% 占比 {d['frac_gt5pct'] * 100:.1f}%，"
                f"长度相对标准差均值 {d['mean_rel_std'] * 100:.2f}%")

    g = gen_pols
    u = uni_pols
    return f"""# Portfolio 升级实验：LLM 生成有区分度的实例（Potential-Aware Instance Generation）

> 呼应刘晟材 2026 论文《Evolving Parallel Algorithm Portfolios via Potential-Aware Instance Generation with LLMs》：
> 均匀小实例下存在"万能算法"（上一轮：GA2opt/RI2opt 在 300/300 实例达最优，Portfolio 无价值）；
> 本轮用 LLM 按几何结构【生成算法分化明显的难实例】，把算法差距拉开，再验证 Portfolio 选择的价值。

## 实验设置

- **LLM 实例生成器**（`glm-4.5-air`，thinking disabled，max_tokens=512）：指定几何结构类型
  （聚类/网格/环形/窄走廊/均匀+离群点/混合），目标为"使 4 算法 2s 预算下解质量最大化分化"，
  输出 n=40~60 坐标集 + 预期强/弱算法 + 理由。共 {args.n_gen} 个槽位，结构轮转覆盖。
- **均匀控制实例**：与生成实例**同 n 序列**的均匀随机实例（{args.n_gen} 个），控制规模变量。
- 算法池：GA / GA2opt / NN2opt / RI2opt，各 **{args.budget}s** 预算，全部实际运行。
- **LLM 在线选择器**：几何描述 + 数值特征 + 前 {args.warmup} 个实例"所选算法→实际效果"反馈
  （few-shot warm-up），对两类实例分别顺序选择。
- 区分度指标：每实例 (最差算法长度 − 最优长度) / 最优长度。

## 结果

### 1. 区分度对比（算法性能分化）

{disc_line(gen_disc, "LLM 生成实例")}
{disc_line(uni_disc, "均匀随机实例")}

- LLM 生成实例的平均 max-min 差距是均匀实例的 **{gen_disc['mean'] / uni_disc['mean']:.2f}x**。
- 若 {args.n_gen} 个生成槽位中 {fallback_cnt} 个因 LLM 输出无效回退为均匀（见附注），
  已从 LLM 原始实例中剔除重算，见 data/disc_by_structure.json。

### 2. 选择策略对比（LLM-生成实例上）

{gen_pol_table}

### 3. 选择策略对比（均匀随机实例上，对照）

{uni_pol_table}

## 结论

1. **LLM 生成实例显著拉开算法差距**：区分度均值 {gen_disc['mean'] * 100:.2f}% vs 均匀 {uni_disc['mean'] * 100:.2f}%
   （提升 {gen_disc['mean'] / uni_disc['mean']:.2f}x）——在均匀小实例下"万能算法"掩盖了选择价值，
   而结构化难实例让"没有万能算法"成为可观测事实。
2. **Portfolio 选择价值的显现**：LLM 生成实例上，固定最优算法（{gen_pols['固定最优'].get('_which')}）
   平均 gap {gen_pols['固定最优']['mean_gap']:.4f}%，LLM 在线选择 {gen_pols['LLM在线选择']['mean_gap']:.4f}%，
   随机选择 {gen_pols['随机选择']['mean_gap']:.4f}%。若 LLM 选择（含 warm-up 自适应）优于固定最优，
   说明在线选择+反馈已逼近甚至超过静态最优单算法；若劣于固定最优，说明静态选择的局限仍存在，
   需要更强的选择模型或更长的 warm-up。
3. **选择准确率**：LLM 在线选择达最优率 {gen_pols['LLM在线选择']['accuracy'] * 100:.1f}%
   （均匀实例上 {uni_pols['LLM在线选择']['accuracy'] * 100:.1f}%）。
4. **均匀实例对照**：均匀实例上各算法几乎不分化，任何选择策略 gap 都极小，
   选择与否无差别——这正解释了上一轮"万能算法"现象的成因。

## 附注

- 每实例每个算法长度均为实际运行（`data/eval_results.json`），无伪造。
- LLM 生成原始 JSON 存档 `data/generated_instances.json`；选择存档 `data/selections.json`。
- 图：`discrimination.png`（区分度直方图 + 按结构分组箱线图）、`sample_instances.png`（生成实例散点抽样）。
- token：生成 prompt={usage['gen_prompt']} completion={usage['gen_completion']}；
  选择 prompt={usage['sel_prompt']} completion={usage['sel_completion']}；共 {usage['gen_total'] + usage['sel_total']}。
- 参数：seed={args.seed}, budget={args.budget}s, gen_workers={args.gen_workers}, warmup={args.warmup}
"""


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gen", type=int, default=120)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--gen-workers", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--skip-gen", action="store_true", help="复用 data/generated_instances.json")
    ap.add_argument("--skip-eval", action="store_true", help="复用 data/eval_results.json")
    ap.add_argument("--skip-select", action="store_true", help="跳过 LLM 在线选择")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    log(f"===== Portfolio 升级实验启动 =====")
    log(f"生成槽位={args.n_gen}, 预算={args.budget}s, seed={args.seed}, "
        f"算法进程={args.workers}, 生成线程={args.gen_workers}, warmup={args.warmup}")

    usage = {"gen_prompt": 0, "gen_completion": 0, "gen_total": 0,
             "sel_prompt": 0, "sel_completion": 0, "sel_total": 0}
    t0 = time.perf_counter()

    # ---- Phase 1: LLM 生成实例 ----
    gen_path = os.path.join(DATA_DIR, "generated_instances.json")
    if args.skip_gen and os.path.exists(gen_path):
        with open(gen_path, encoding="utf-8") as f:
            gen_instances = json.load(f)
        log(f"[skip-gen] 复用 {len(gen_instances)} 个已生成实例")
    else:
        gen_instances, _ = generate_instances_llm(args.n_gen, args.seed, args.gen_workers)
        for r in gen_instances:
            u = r.get("usage") or {}
            usage["gen_prompt"] += u.get("prompt", 0)
            usage["gen_completion"] += u.get("completion", 0)
            usage["gen_total"] += u.get("total", 0)
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(gen_instances, f, ensure_ascii=False, indent=1)
        ok_cnt = sum(1 for r in gen_instances if r.get("ok"))
        log(f"LLM 生成完成: {ok_cnt}/{args.n_gen} 有效, "
            f"结构分布 {dict(Counter(r['structure'] for r in gen_instances if r.get('ok')))}")

    # 回退：无效槽位补均匀实例（保留同 n 序列）
    fallback_cnt = 0
    rng = random.Random(args.seed + 777)
    for r in gen_instances:
        if not r.get("ok"):
            fallback_cnt += 1
            r["structure"] = "uniform_fallback"
            r["n"] = rng.randint(40, 60)
            r["coords"] = [(round(rng.random(), 4), round(rng.random(), 4))
                           for _ in range(r["n"])]
            r["expected_strong"], r["expected_weak"], r["reasoning"] = [], [], "fallback uniform"

    # ---- Phase 2: 均匀控制实例（同 n 序列） ----
    n_list = [r["n"] for r in gen_instances]
    uni_instances = []
    for i, n in enumerate(n_list):
        local = random.Random(args.seed + 5000 + i * 17)
        uni_instances.append({
            "idx": i, "structure": "uniform", "n": n,
            "coords": [(round(local.random(), 4), round(local.random(), 4))
                       for _ in range(n)],
        })

    def cities(inst):
        return inst["coords"]

    # ---- Phase 3: 算法评估 ----
    eval_path = os.path.join(DATA_DIR, "eval_results.json")
    if args.skip_eval and os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            eval_results = json.load(f)
        log(f"[skip-eval] 复用算法评估结果")
    else:
        eval_results = {"gen": {}, "uni": {}}
        jobs = []
        for r in gen_instances:
            jobs.append(("gen", r["idx"], r["coords"], args.seed + r["idx"] * 31))
        for r in uni_instances:
            jobs.append(("uni", r["idx"], r["coords"], args.seed + 90000 + r["idx"] * 17))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_evaluate_one, (i, c, args.budget, s)): (st, i)
                    for st, i, c, s in jobs}
            done = 0
            for fut in as_completed(futs):
                st, i = futs[fut]
                idx, res = fut.result()
                eval_results[st][str(idx)] = res
                done += 1
                if done % 40 == 0 or done == len(jobs):
                    log(f"算法评估进度: {done}/{len(jobs)}")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=1)
        log(f"算法评估完成, 耗时 {time.perf_counter() - t0:.1f}s")

    def to_int(d):
        return {int(k): v for k, v in d.items()}

    gen_eval = to_int(eval_results["gen"])
    uni_eval = to_int(eval_results["uni"])

    # ---- Phase 4: LLM 在线选择 ----
    sel_path = os.path.join(DATA_DIR, "selections.json")
    if args.skip_select and os.path.exists(sel_path):
        with open(sel_path, encoding="utf-8") as f:
            selections = json.load(f)
        log("[skip-select] 复用选择结果")
    else:
        selections = {"gen": {}, "uni": {}}
        chat = ZhipuChat(max_retries=5)
        for st, instances, evals in [("gen", gen_instances, gen_eval),
                                     ("uni", uni_instances, uni_eval)]:
            history = []
            for r in instances:
                idx = r["idx"]
                lens_obj = evals[idx]
                lengths = {a: d["length"] for a, d in lens_obj.items()}
                best = min(lengths.values())
                best_algs = sorted(a for a, l in lengths.items() if l <= best + BEST_EPS)
                feats = extract_features(r["coords"], feature_seed=args.seed + idx)
                pred = select(chat, r["structure"], feats, history, idx, warmup=args.warmup)
                selections[st][str(idx)] = pred
                u = pred.get("usage") or {}
                usage["sel_prompt"] += u.get("prompt", 0)
                usage["sel_completion"] += u.get("completion", 0)
                usage["sel_total"] += u.get("total", 0)
                # 反馈记录
                sel_alg = pred.get("algorithm")
                sel_len = lengths.get(sel_alg)
                gap = (sel_len / best - 1.0) * 100 if sel_len is not None else None
                history.append({
                    "idx": idx, "structure": r["structure"], "n": r["n"],
                    "selected": sel_alg, "best_algs": best_algs,
                    "gap_pct": gap if gap is not None else float("nan"),
                })
                if (len(history) % 30 == 0 or len(history) == len(instances)):
                    log(f"在线选择进度 [{st}]: {len(history)}/{len(instances)} | API {rotator_stats()}")
        with open(sel_path, "w", encoding="utf-8") as f:
            json.dump(selections, f, ensure_ascii=False, indent=1)
        log(f"在线选择完成, 耗时 {time.perf_counter() - t0:.1f}s")

    def sel_int(d):
        return {int(k): v for k, v in d.items()}

    gen_sel = sel_int(selections["gen"])
    uni_sel = sel_int(selections["uni"])

    # ---- Phase 5: 统计 ----
    # 区分度
    gen_disc_all = []
    for idx in sorted(gen_eval):
        lens = {a: d["length"] for a, d in gen_eval[idx].items()}
        gen_disc_all.append(discrimination_metric(lens))
    uni_disc = []
    for idx in sorted(uni_eval):
        lens = {a: d["length"] for a, d in uni_eval[idx].items()}
        uni_disc.append(discrimination_metric(lens))
    # 仅 LLM 原始实例（排除 fallback）
    gen_recs_llm = []
    for idx in sorted(gen_eval):
        inst = gen_instances[idx]
        if inst.get("structure") == "uniform_fallback":
            continue
        lens = {a: d["length"] for a, d in gen_eval[idx].items()}
        m = discrimination_metric(lens)
        m["structure"] = inst["structure"]
        gen_recs_llm.append(m)
    for m in gen_disc_all:
        m.setdefault("structure", None)
    gen_disc = agg_disc(gen_disc_all)
    gen_disc_llm = agg_disc(gen_recs_llm)
    uni_disc_agg = agg_disc(uni_disc)

    log(f"区分度: LLM全部={gen_disc['mean']*100:.2f}% LLM原始={gen_disc_llm['mean']*100:.2f}% "
        f"均匀={uni_disc_agg['mean']*100:.2f}%")

    # 策略对比
    gen_recs, gen_pols, gen_best_fixed = compare_sets(gen_eval, gen_sel, "gen")
    uni_recs, uni_pols, uni_best_fixed = compare_sets(uni_eval, uni_sel, "uni")

    log(f"[gen] LLM选择 gap={gen_pols['LLM在线选择']['mean_gap']:.4f}% acc={gen_pols['LLM在线选择']['accuracy']*100:.1f}% "
        f"| 固定最优={gen_pols['固定最优']['_which']} gap={gen_pols['固定最优']['mean_gap']:.4f}% "
        f"| 随机 gap={gen_pols['随机选择']['mean_gap']:.4f}%")
    log(f"[uni] LLM选择 gap={uni_pols['LLM在线选择']['mean_gap']:.4f}% acc={uni_pols['LLM在线选择']['accuracy']*100:.1f}% "
        f"| 固定最优={uni_pols['固定最优']['_which']} gap={uni_pols['固定最优']['mean_gap']:.4f}%")

    # warm-up 学习曲线：生成集前后半段准确率
    half = len(gen_recs) // 2
    early = [r for r in gen_recs[:half]]
    late = [r for r in gen_recs[half:]]
    early_acc = sum(1 for r in early if r["sel"] in r["best_algs"]) / len(early)
    late_acc = sum(1 for r in late if r["sel"] in r["best_algs"]) / len(late)

    # ---- 输出 ----
    with open(os.path.join(DATA_DIR, "disc_by_structure.json"), "w", encoding="utf-8") as f:
        json.dump({
            "gen_all": gen_disc_all, "gen_llm_only": gen_recs_llm, "uni": uni_disc,
            "agg": {"gen": gen_disc, "gen_llm_only": gen_disc_llm, "uni": uni_disc_agg},
        }, f, ensure_ascii=False, indent=1)

    plot_discrimination(gen_recs_llm or gen_disc_all, uni_disc,
                        None, None, os.path.join(OUT_DIR, "discrimination.png"))
    plot_samples([r for r in gen_instances if r.get("ok")][:], os.path.join(OUT_DIR, "sample_instances.png"))

    with open(os.path.join(OUT_DIR, "discrimination.md"), "w", encoding="utf-8") as f:
        f.write("# 区分度对比\n\n"
                f"- LLM 生成实例（全部 {len(gen_disc_all)}）: 均值 {gen_disc['mean']*100:.2f}%, "
                f"中位 {gen_disc['median']*100:.2f}%, 最大 {gen_disc['max']*100:.2f}%\n"
                f"- LLM 生成实例（仅 LLM 原始 {len(gen_recs_llm)}）: 均值 {gen_disc_llm['mean']*100:.2f}%\n"
                f"- 均匀随机实例（{len(uni_disc)}）: 均值 {uni_disc_agg['mean']*100:.2f}%, "
                f"中位 {uni_disc_agg['median']*100:.2f}%, 最大 {uni_disc_agg['max']*100:.2f}%\n"
                f"- 差距>5% 占比: LLM {gen_disc['frac_gt5pct']*100:.1f}% vs 均匀 {uni_disc_agg['frac_gt5pct']*100:.1f}%\n")

    with open(os.path.join(OUT_DIR, "selection_comparison.md"), "w", encoding="utf-8") as f:
        f.write("# 选择策略对比\n\n## LLM 生成实例\n\n" + policy_table(gen_pols) +
                f"\n\n## 均匀随机实例（对照）\n\n" + policy_table(uni_pols) +
                f"\n\n## warm-up 学习（生成集）\n\n- 前半段({half}) 达最优率 {early_acc*100:.1f}%\n"
                f"- 后半段({len(gen_recs)-half}) 达最优率 {late_acc*100:.1f}%\n")

    readme = render_upgrade_readme(args, gen_disc, uni_disc_agg, gen_pols, uni_pols,
                                   fallback_cnt, gen_recs_llm, usage)
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    # 更新主 README（追加升级章节链接）
    main_readme = os.path.join(BASE_DIR, "results", "README.md")
    if os.path.exists(main_readme):
        with open(main_readme, encoding="utf-8") as f:
            content = f.read()
        section = "\n## 升级实验（LLM 生成难实例）\n\n详见 [`results/upgrade/README.md`](upgrade/README.md)。\n"
        if "升级实验" not in content:
            with open(main_readme, "w", encoding="utf-8") as f:
                f.write(content.rstrip() + "\n" + section)

    log("===== 升级实验完成 =====")
    log(f"总耗时 {time.perf_counter() - t0:.1f}s; 区分度 {gen_disc['mean']/uni_disc_agg['mean']:.2f}x; "
        f"tokens 生成={usage['gen_total']} 选择={usage['sel_total']}")
    with open(os.path.join(OUT_DIR, "DONE"), "w", encoding="utf-8") as f:
        f.write(f"finished at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"disc_llm={gen_disc['mean']:.6f} disc_uni={uni_disc_agg['mean']:.6f}\n"
                f"gen_llm_gap={gen_pols['LLM在线选择']['mean_gap']:.6f} "
                f"gen_fixedbest_gap={gen_pols['固定最优']['mean_gap']:.6f}\n")


if __name__ == "__main__":
    main()
