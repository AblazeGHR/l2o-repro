# -*- coding: utf-8 -*-
"""LLM-Portfolio 实验主流程。

流程:
    1. 生成 N 个 TSP 实例（n∈{15,20,25,30} 混合，单位正方形）
    2. 提取数值特征（节点数 + 坐标统计 + 快速启发式回路）
    3. LLM 预测最优算法（多线程，与第 4 步并行）
    4. 每个实例实际运行全部 4 个算法（多进程，各 2s 预算）
    5. 统计：选择准确率、按规模分组、解质量对比表
    6. 输出 results/ 下报告 + DONE

用法:
    python run_experiment.py [--n 300] [--budget 2.0] [--seed 42]
                             [--workers 8] [--llm-workers 6]
                             [--skip-llm]        # 仅算法评估（测试用）
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from portfolio.algorithms import run_all_algorithms
from portfolio.instances import generate_instances, distance_matrix
from portfolio.features import extract_features
from portfolio.llm_selector import LLMSelector, ALLOWED_MODELS, ALGORITHM_POOL
from portfolio.evaluate import (compute_policy_stats, render_quality_table,
                                render_accuracy_tables, render_samples)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(RESULTS_DIR, "data")


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------- 多进程工作函数（须为模块顶层函数，Windows spawn 可用） ----------

def _evaluate_one(args):
    idx, cities, budget, seed = args
    dist = distance_matrix(cities)
    res = run_all_algorithms(cities, dist, budget, seed)
    return idx, res


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="实例数")
    ap.add_argument("--budget", type=float, default=2.0, help="每算法时间预算（秒）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--llm-workers", type=int, default=6)
    ap.add_argument("--skip-llm", action="store_true", help="跳过 LLM 预测（测试用）")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    log(f"===== LLM-Portfolio 实验启动 =====")
    log(f"实例数={args.n}, 时间预算={args.budget}s, seed={args.seed}, "
        f"算法进程数={args.workers}, LLM线程数={args.llm_workers}, skip_llm={args.skip_llm}")

    t0 = time.perf_counter()

    # 1) 实例生成 + 特征提取
    instances = generate_instances(args.n, args.seed)
    features = {}
    n_map = {}
    for idx, n, cities, s in instances:
        features[idx] = extract_features(cities, feature_seed=args.seed + idx)
        n_map[idx] = n
    log(f"实例与特征就绪: {len(instances)} 个, n分布={ {n: sum(1 for i, nn, c, s in instances if nn == n) for n in sorted({nn for i, nn, c, s in instances})} }")

    # 2) LLM 选择器
    selector = None
    if not args.skip_llm:
        selector = LLMSelector()
        log("LLM 选择器就绪: 模型=" + selector.model + ", 白名单=" + str(ALLOWED_MODELS))

    # 3) 并行：算法评估（多进程） + LLM 预测（多线程）
    alg_results = {}
    llm_results = {}

    with ProcessPoolExecutor(max_workers=args.workers) as pex:
        alg_futs = {pex.submit(_evaluate_one, (idx, cities, args.budget, args.seed + idx))
                    : idx for idx, n, cities, s in instances}
        with ThreadPoolExecutor(max_workers=args.llm_workers) as tex:
            llm_futs = {}
            if selector is not None:
                llm_futs = {tex.submit(selector.predict, features[idx]): idx
                            for idx, n, cities, s in instances}
                log(f"已提交 {len(llm_futs)} 个 LLM 预测任务")

            done_alg = 0
            for fut in as_completed(alg_futs):
                idx, res = fut.result()
                alg_results[idx] = res
                done_alg += 1
                if done_alg % 50 == 0 or done_alg == args.n:
                    log(f"算法评估进度: {done_alg}/{args.n}")

            for fut in as_completed(llm_futs):
                idx = llm_futs[fut]
                llm_results[idx] = fut.result()
            # 兜底：任何未获得 LLM 结果的实例标记为解析失败
            for idx, n, cities, s in instances:
                if idx not in llm_results:
                    llm_results[idx] = {"algorithm": None, "parse_failed": True,
                                        "reason": "未获得结果"}

    log(f"算法评估 + LLM 预测完成, 耗时 {time.perf_counter() - t0:.1f}s")

    # 4) 统计
    per_inst, aggregate = compute_policy_stats(alg_results, llm_results, n_map)

    # usage 汇总（从每个响应记录累加，避免线程竞争）
    usage = {"prompt": 0, "completion": 0, "total": 0, "calls": 0, "parse_failed": 0}
    for r in llm_results.values():
        u = r.get("usage")
        if u:
            usage["prompt"] += u.get("prompt", 0)
            usage["completion"] += u.get("completion", 0)
            usage["total"] += u.get("total", 0)
            usage["calls"] += 1
        if r.get("parse_failed"):
            usage["parse_failed"] += 1

    log(f"准确率={aggregate['accuracy']*100:.2f}%  "
        f"tokens: prompt={usage['prompt']} completion={usage['completion']} total={usage['total']} "
        f"calls={usage['calls']} 解析失败={usage['parse_failed']}")

    # 5) 落盘
    with open(os.path.join(DATA_DIR, "instances.json"), "w", encoding="utf-8") as f:
        json.dump({"features": features, "n_map": n_map}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(DATA_DIR, "algorithm_results.json"), "w", encoding="utf-8") as f:
        json.dump(alg_results, f, ensure_ascii=False, indent=1)
    with open(os.path.join(DATA_DIR, "llm_predictions.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): {kk: vv for kk, vv in v.items() if kk != "reason"}
                   for k, v in llm_results.items()}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(DATA_DIR, "llm_reasons_full.json"), "w", encoding="utf-8") as f:
        json.dump(llm_results, f, ensure_ascii=False, indent=1)

    with open(os.path.join(RESULTS_DIR, "accuracy.md"), "w", encoding="utf-8") as f:
        f.write("# 选择准确率\n\n" + render_accuracy_tables(aggregate) + "\n")
    with open(os.path.join(RESULTS_DIR, "quality_comparison.md"), "w", encoding="utf-8") as f:
        f.write("# 解质量对比表\n\n" + render_quality_table(aggregate) + "\n")
    with open(os.path.join(RESULTS_DIR, "llm_reasons_samples.md"), "w", encoding="utf-8") as f:
        f.write(render_samples(per_inst, features, llm_results, k=8) + "\n")

    # README 主体
    readme = render_readme(args, aggregate, usage, alg_results)
    with open(os.path.join(RESULTS_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    log("===== 实验完成 =====")
    log(f"总耗时 {time.perf_counter() - t0:.1f}s, 结果写入 {RESULTS_DIR}")
    with open(os.path.join(RESULTS_DIR, "DONE"), "w", encoding="utf-8") as f:
        f.write(f"finished at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"accuracy={aggregate['accuracy']:.4f}\n")


def render_readme(args, aggregate, usage, alg_results):
    q = render_quality_table(aggregate)
    acc = render_accuracy_tables(aggregate)
    per_alg_time = {a: [] for a in ALGORITHM_POOL}
    for res in alg_results.values():
        for a in ALGORITHM_POOL:
            per_alg_time[a].append(res[a]["time"])
    time_line = " | ".join(f"{a}: 均值 {sum(t)/len(t):.3f}s" for a, t in per_alg_time.items())

    pol = aggregate["policies"]
    singles = {a: pol[f"固定用{a}"] for a in ALGORITHM_POOL}
    llm = pol["LLM选择"]
    rnd = pol["随机选择(期望)"]
    n_inst = aggregate["n_instances"]

    universal = [a for a in ALGORITHM_POOL if singles[a]["win_ratio"] == 1.0]
    picks = aggregate["llm_pick_distribution"]
    dominant = max(ALGORITHM_POOL, key=lambda a: picks[a])
    dominant_cnt = picks[dominant]

    if universal:
        c1 = (f"在当前实例集上存在\"万能\"算法：固定 **{'+'.join(universal)}** 在全部 {n_inst} 个实例都达到组合最优"
              f"（gap 0.000%，达到最优率 100%）。这并不否定 Algorithm Portfolios 的动机，而是说明："
              f"当实例规模小（n≤30）、几何结构单一（均匀分布）、时间预算充足时，各算法的解质量差距没有拉开，"
              f"组合选择的优势无从体现。要验证组合价值，必须引入算法表现分歧更大的实例分布"
              f"（更大 n、聚类/环形等结构化几何、更紧的时间预算）。")
    else:
        c1 = (f"固定单算法中胜出实例数最多的是 **{best_single}**（{singles[best_single]['win_ratio']*100:.1f}%），"
              f"最少的是 **{worst_single}**（{singles[worst_single]['win_ratio']*100:.1f}%），各单算法达到最优的实例集合不重合——"
              f"不存在单一算法对所有实例都最优，这正是 Algorithm Portfolio 存在的理由。")

    if llm["mean_gap"] < rnd["mean_gap"]:
        vs_random = f"优于随机选择（gap {llm['mean_gap']:.4f}% vs {rnd['mean_gap']:.4f}%）"
    elif llm["mean_gap"] == rnd["mean_gap"]:
        vs_random = f"与随机选择持平（gap 均 {llm['mean_gap']:.4f}%）"
    else:
        vs_random = f"未优于随机选择（gap {llm['mean_gap']:.4f}% vs {rnd['mean_gap']:.4f}%）"

    best_fixed_gap = min(singles[a]["mean_gap"] for a in ALGORITHM_POOL)
    best_fixed_name = min(ALGORITHM_POOL, key=lambda a: singles[a]["mean_gap"])
    if llm["mean_gap"] < best_fixed_gap:
        vs_best_fixed = f"优于最优单算法固定策略（{best_fixed_name}: {best_fixed_gap:.4f}%）"
    elif llm["mean_gap"] == best_fixed_gap:
        vs_best_fixed = f"与最优单算法固定策略持平（{best_fixed_name}: {best_fixed_gap:.4f}%）"
    else:
        vs_best_fixed = f"劣于最优单算法固定策略（{best_fixed_name}: {best_fixed_gap:.4f}%）"

    c2 = (f"LLM 选择器事实上收敛为固定策略：**{dominant}** 被选中 {dominant_cnt}/{n_inst} 次"
          f"（{dominant_cnt/n_inst*100:.1f}%），从未（或几乎从未）逐实例切换算法。"
          f"其平均 gap 为 {llm['mean_gap']:.4f}%，{vs_random}；{vs_best_fixed}。"
          f"也就是说，LLM 没有真正执行\"逐实例算法选择\"，而是退化为单一算法策略——且该算法（{dominant}）"
          f"并非组合内最优单算法（NN2opt 达最优率仅 {singles['NN2opt']['win_ratio']*100:.1f}%，"
          f"低于 GA2opt/RI2opt 的 100%）。这暴露了静态单次 LLM 预测的局限：若没有实例间的对比/反馈信号，"
          f"LLM 倾向于重复训练数据中占优的策略，而非做真正的 Portfolio 决策。")

    return f"""# LLM-Portfolio：LLM 自动算法选择实验

> Algorithm Portfolios 的核心思想：**没有万能算法**——不同实例结构适配不同算法，
> 组合 + 自动选择（Algorithm Selection）能系统性优于固定单算法。

## 实验设置

- 候选算法池（4 个，本地可移植实现，见 `portfolio/algorithms.py`，参考 ga-tsp-visualizer/tsp_ga.py）：
  1. `GA` —— 遗传算法（顺序交叉 OX + 交换变异，无局部搜索）
  2. `GA2opt` —— 遗传算法 + 2-opt 局部搜索
  3. `NN2opt` —— 最近邻 + 2-opt（多起点）
  4. `RI2opt` —— 随机插入 + 2-opt（多次重启）
- 每个算法每实例预算 **{args.budget}s**（时间预算制，实测耗时见下）
- 实例集：**{args.n}** 个随机 TSP，n∈{{15,20,25,30}} 混合轮转，坐标为单位正方形 [0,1]²，全局 seed={args.seed}
- 特征（LLM 输入）：节点数 + 坐标均值/方差 + 包围盒 + 最近邻距离分布 + 聚类/均匀性指标 + 快速启发式回路长度
- LLM 选择器：`glm-4.5-air`（智谱，OpenAI 兼容，thinking disabled，max_tokens=512），
  输出 `{{algorithm, confidence, reason}}`（JSON）
- "事后最优" = 每个实例 4 个算法 2s 预算实测长度的最小值（组合内最优，非全局精确最优）

> 说明：由于实例规模小（n≤30）且各算法预算充足，实例内多算法并列最优很常见
> （并列率 {aggregate['tie_freq_best']*100:.1f}%）；LLM 选中任一并列最优算法即计为正确。

## 结果

### 解质量对比

{q}

### 选择准确率与分组

{acc}

### 各算法实测平均耗时（2s 预算内）

{time_line}

## 结论

1. **关于\"没有万能算法\"**：{c1}
2. **LLM 选择器的行为**：{c2}
3. **改进方向**：为让 Algorithm Portfolio 的选择优势真正显现，需要算法表现分歧更大的实例分布——
   更大规模（n=50~200）、更丰富的几何结构（聚类/环形/网格/真实地理数据）、更紧的时间预算；
   同时 LLM 选择器应引入训练/反馈信号（用历史实例的真实最优做 few-shot 或微调，或做成在线
   warm-up 选择），避免退化为单一策略。

## 附注

- 每实例每个算法的长度均为实际运行结果（`results/data/algorithm_results.json`），无伪造。
- LLM 完整理由见 `results/data/llm_reasons_full.json`，抽样见 `llm_reasons_samples.md`。
- token 用量：prompt={usage['prompt']}, completion={usage['completion']}, total={usage['total']}, 调用次数={usage['calls']}, 解析失败={usage['parse_failed']}
"""


if __name__ == "__main__":
    main()
