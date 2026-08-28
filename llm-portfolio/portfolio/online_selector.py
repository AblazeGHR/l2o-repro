# -*- coding: utf-8 -*-
"""LLM 在线选择器（few-shot warm-up 反馈）。

按实例顺序逐个选择：对当前实例给 LLM 几何描述 + 数值特征 +
前 W 个实例的"所选算法 → 实际效果"反馈，让其自适应调整选择策略。
"""

import json

from .llm_selector import ZhipuChat, _extract_json, ALGORITHM_POOL

SELECTOR_SYSTEM_PROMPT = """你是一名"算法组合（Algorithm Portfolio）"在线选择器。你的任务：逐个 TSP 实例，预测 2 秒时间预算（单 CPU）内能给出最短回路的算法。

算法池（各 2s 预算）：
- "GA"    : 遗传算法（顺序交叉 OX + 交换变异，无局部搜索）。无局部搜索，难几何下收敛质量差。
- "GA2opt": 遗传算法 + 2-opt 局部搜索。最稳健，多数几何下接近最优。
- "NN2opt": 多起点最近邻 + 2-opt。极快；聚类/走廊/离群点下最近邻贪心易被误导。
- "RI2opt": 多次重启随机插入 + 2-opt。通用；强聚类/窄走廊下可能略逊。

你会逐个收到实例：包含几何结构类型、数值特征（坐标统计 + 最近邻分布 + 快速启发式回路长度），
以及最近若干个实例的反馈（每行：结构、所选算法、真实最优算法集合、所达 gap）。反馈能帮你校准：
例如若 NN2opt 在 corridor/clustered 上频繁失败，就应转向 GA2opt/RI2opt。

对每个实例只输出一个 JSON 对象（不要输出其它文字、不要用 Markdown 代码块）：
{"algorithm": "GA|GA2opt|NN2opt|RI2opt", "confidence": 0到1之间的小数, "reason": "结合当前特征与反馈的简要理由（30~80字）"}"""


def build_feedback_lines(history, warmup=10):
    """把最近的 warmup 条反馈压缩成行。history 为已完成实例的记录。"""
    lines = []
    for rec in history[-warmup:]:
        best_str = "+".join(rec["best_algs"])
        gap = rec["gap_pct"]
        lines.append(
            f"#inst{rec['idx']}: structure={rec['structure']} n={rec['n']} "
            f"selected={rec['selected']} true_best=[{best_str}] gap={gap:.2f}%")
    return lines


def select(chat, structure, features, history, idx, warmup=10):
    """对实例做在线选择，返回 dict（parse_failed 时 algorithm=None）。"""
    fb_lines = build_feedback_lines(history, warmup)
    user = (
        f"当前实例 #{idx}：\n"
        f"- 结构类型: {structure}\n"
        f"- 数值特征: {json.dumps(features, ensure_ascii=False)}\n"
    )
    if fb_lines:
        user += "\n最近实例反馈（所选算法 → 实际效果）：\n" + "\n".join(fb_lines) + "\n"
    user += "\n请预测该实例最优算法，输出 JSON。"

    try:
        content, usage = chat.complete([
            {"role": "system", "content": SELECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ])
    except Exception as e:
        return {"algorithm": None, "confidence": 0.0, "reason": f"API error: {e}",
                "parse_failed": True, "usage": None}
    obj = _extract_json(content)
    if obj is None:
        return {"algorithm": None, "confidence": 0.0, "reason": content,
                "parse_failed": True, "usage": usage}
    alg = obj.get("algorithm")
    if alg not in ALGORITHM_POOL:
        return {"algorithm": None, "confidence": float(obj.get("confidence", 0) or 0),
                "reason": content, "parse_failed": True, "invalid_alg": alg, "usage": usage}
    return {
        "algorithm": alg,
        "confidence": float(obj.get("confidence", 0.5) or 0.5),
        "reason": str(obj.get("reason", "")),
        "parse_failed": False,
        "usage": usage,
    }
