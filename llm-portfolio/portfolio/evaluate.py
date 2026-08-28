# -*- coding: utf-8 -*-
"""评估与统计：从算法结果 + LLM 预测计算准确率、解质量对比，并生成 Markdown 报告。"""

import json

from .llm_selector import ALGORITHM_POOL

BEST_EPS = 1e-9


def compute_policy_stats(alg_results, llm_results, n_map):
    """计算整体统计。

    alg_results: {idx: {alg: {"length": float, ...}}}
    llm_results: {idx: {algorithm: str|None, parse_failed: bool, ...}}
    n_map:       {idx: n}
    返回 (per_instance_stats, aggregate)。
    """
    per_inst = []
    llm_pick_counts = {a: 0 for a in ALGORITHM_POOL}
    llm_parse_fail = 0

    for idx, lengths_obj in sorted(alg_results.items()):
        lengths = {a: d["length"] for a, d in lengths_obj.items()}
        best_len = min(lengths.values())
        best_algs = {a for a, l in lengths.items() if l <= best_len + BEST_EPS}

        n = n_map[idx]
        sel = llm_results[idx]["algorithm"]
        if sel is None:
            llm_parse_fail += 1
        else:
            llm_pick_counts[sel] = llm_pick_counts.get(sel, 0) + 1

        # 随机选择的期望值（均匀随机从 4 个算法中挑一个）
        rand_expected_len = sum(lengths.values()) / len(lengths)
        rand_expected_win = len(best_algs) / len(lengths)

        # 解析失败的实例：质量上按"随机选择"兜底（可操作口径），准确率上计为不正确
        sel_len = lengths.get(sel) if sel else None
        sel_correct = (sel in best_algs) if sel else False
        sel_gap = (sel_len / best_len - 1.0) if sel_len is not None else None
        eff_len = sel_len if sel_len is not None else rand_expected_len
        eff_gap = sel_gap if sel_gap is not None else (rand_expected_len / best_len - 1.0)
        eff_win = sel_correct if sel else rand_expected_win

        per_inst.append({
            "idx": idx, "n": n,
            "lengths": lengths,
            "best_len": best_len,
            "best_algs": sorted(best_algs),
            "selected": sel,
            "selected_len": sel_len,
            "selected_correct": sel_correct,
            "selected_gap": sel_gap,
            "rand_expected_len": rand_expected_len,
            "rand_expected_win": rand_expected_win,
            # 有效口径（含兜底）
            "eff_len": eff_len,
            "eff_gap": eff_gap,
            "eff_win": eff_win,
        })

    M = len(per_inst)

    # 策略汇总
    def summarize(key_fn, win_fn, gap_fn):
        lens = [key_fn(p) for p in per_inst]
        wins = sum(win_fn(p) for p in per_inst)
        gaps = [gap_fn(p) for p in per_inst if gap_fn(p) is not None]
        return {
            "mean_len": sum(lens) / len(lens),
            "mean_gap": (sum(gaps) / len(gaps)) * 100 if gaps else None,
            "wins": wins,
            "win_ratio": wins / M,
        }

    policies = {}
    policies["LLM选择"] = summarize(
        lambda p: p["eff_len"], lambda p: p["eff_win"],
        lambda p: p["eff_gap"])
    policies["随机选择(期望)"] = summarize(
        lambda p: p["rand_expected_len"], lambda p: p["rand_expected_win"],
        lambda p: p["rand_expected_len"] / p["best_len"] - 1.0)
    policies["事后最优(oracle)"] = summarize(
        lambda p: p["best_len"], lambda p: True,
        lambda p: 0.0)
    for a in ALGORITHM_POOL:
        policies[f"固定用{a}"] = summarize(
            lambda p, a=a: p["lengths"][a], lambda p, a=a: a in p["best_algs"],
            lambda p, a=a: p["lengths"][a] / p["best_len"] - 1.0)

    # 按规模分组准确率
    by_size = {}
    for p in per_inst:
        d = by_size.setdefault(p["n"], {"total": 0, "correct": 0})
        d["total"] += 1
        d["correct"] += int(p["selected_correct"])
    by_size = {n: {"total": d["total"], "correct": d["correct"],
                   "accuracy": d["correct"] / d["total"]}
               for n, d in sorted(by_size.items())}

    # 头部对比：LLM 选择 vs 各单算法（胜/平/负）
    head2head = {}
    for a in ALGORITHM_POOL:
        win = tie = loss = 0
        for p in per_inst:
            ls = p["selected_len"]
            if ls is None:
                continue
            la = p["lengths"][a]
            if ls < la - BEST_EPS:
                win += 1
            elif ls > la + BEST_EPS:
                loss += 1
            else:
                tie += 1
        head2head[a] = {"win": win, "tie": tie, "loss": loss}

    aggregate = {
        "n_instances": M,
        "accuracy": sum(p["selected_correct"] for p in per_inst) / M,
        "parse_fail_rate": llm_parse_fail / M,
        "llm_pick_distribution": llm_pick_counts,
        "by_size": by_size,
        "policies": policies,
        "head2head": head2head,
        "tie_freq_best": sum(len(p["best_algs"]) > 1 for p in per_inst) / M,
        "mean_selected_gap": (sum(p["selected_gap"] for p in per_inst if p["selected_gap"] is not None)
                              / M) * 100,
    }
    return per_inst, aggregate


def _fmt_policy(name, s):
    return (f"| {name} | {s['mean_len']:.5f} | "
            f"{s['mean_gap']:.3f}%" if s['mean_gap'] is not None else "-") + \
           f" | {s['wins']} | {s['win_ratio'] * 100:.1f}% |"


def render_quality_table(aggregate):
    lines = [
        "| 策略 | 平均回路长度 | 平均gap(vs最优) | 达到最优实例数 | 达到最优比例 |",
        "|---|---|---|---|---|",
    ]
    pol = aggregate["policies"]
    order = ["LLM选择", "随机选择(期望)", "事后最优(oracle)"] + \
            [f"固定用{a}" for a in ALGORITHM_POOL]
    for name in order:
        lines.append(_fmt_policy(name, pol[name]))
    return "\n".join(lines)


def render_accuracy_tables(aggregate):
    a = aggregate
    out = []
    out.append(f"- 总实例数: {a['n_instances']}")
    out.append(f"- **总体选择准确率: {a['accuracy'] * 100:.2f}%**")
    out.append(f"- LLM 输出解析失败率: {a['parse_fail_rate'] * 100:.2f}%")
    out.append(f"- 实例平均存在多个并列最优算法（平局）比例: {a['tie_freq_best'] * 100:.1f}%")
    out.append(f"- LLM 所选算法平均 gap(vs 最优): {a['mean_selected_gap']:.4f}%")
    out.append("")
    out.append("**按实例规模分组准确率**")
    out.append("| n | 实例数 | 正确数 | 准确率 |")
    out.append("|---|---|---|---|")
    for n, d in a["by_size"].items():
        out.append(f"| {n} | {d['total']} | {d['correct']} | {d['accuracy'] * 100:.2f}% |")
    out.append("")
    out.append("**LLM 选择分布**")
    for alg, cnt in sorted(a["llm_pick_distribution"].items()):
        out.append(f"- {alg}: {cnt} 次 ({cnt / a['n_instances'] * 100:.1f}%)")
    out.append("")
    out.append("**头部对比：LLM 选择 vs 各单算法（胜/平/负）**")
    out.append("| 对比对象 | 胜(LLM更短) | 平 | 负(LLM更长) |")
    out.append("|---|---|---|---|")
    for alg, h in a["head2head"].items():
        out.append(f"| vs {alg} | {h['win']} | {h['tie']} | {h['loss']} |")
    return "\n".join(out)


def render_samples(per_inst, features, llm_results, k=8):
    """LLM 理由抽样存档：均匀抽样 k 个实例展示特征/预测/理由/真实结果。"""
    M = len(per_inst)
    step = max(1, M // k)
    idxs = [p["idx"] for p in per_inst[::step][:k]]
    lines = [f"# LLM 理由抽样存档（{len(idxs)}/{M} 实例）", ""]
    for idx in idxs:
        p = next(x for x in per_inst if x["idx"] == idx)
        llm = llm_results[idx]
        feats = features[idx]
        lines.append(f"## 实例 #{idx}（n={p['n']}）")
        lines.append("")
        lines.append("**特征**")
        lines.append("```json")
        lines.append(json.dumps(feats, ensure_ascii=False, indent=1))
        lines.append("```")
        lines.append("")
        lines.append("**LLM 预测**")
        lines.append(f"- 算法: {llm.get('algorithm')}")
        lines.append(f"- 置信度: {llm.get('confidence')}")
        lines.append(f"- 解析失败: {llm.get('parse_failed')}")
        lines.append("")
        lines.append("**理由**")
        lines.append("> " + (llm.get("reason") or "").replace("\n", "\n> "))
        lines.append("")
        lines.append("**真实结果（各算法 2s 预算实测长度）**")
        lines.append("| 算法 | 长度 |")
        lines.append("|---|---|")
        for a in ALGORITHM_POOL:
            lines.append(f"| {a} | {p['lengths'][a]:.5f} |")
        lines.append(f"| **最优** | **{p['best_len']:.5f}** |")
        lines.append(f"| LLM 所选 gap | {p['selected_gap'] * 100:.3f}% |" if p["selected_gap"] is not None
                     else "| LLM 所选 gap | -（解析失败） |")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)
