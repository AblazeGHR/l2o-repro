# -*- coding: utf-8 -*-
"""LLM 实例生成器：生成"算法性能分化"的 TSP 实例。

给 LLM 指定几何结构类型 + 分化目标，让其输出 n=40~60 的坐标集（JSON），
并说明预期强/弱算法。输出经严格校验，无效则重试，仍失败则回退均匀实例。
"""

import json
import random
import threading

from .llm_selector import ZhipuChat, _extract_json

# 生成器需输出完整坐标集（40~60 点），512 tokens 会被截断，故单独放宽；
# 在线选择器仍保持 max_tokens=512。
GENERATOR_MAX_TOKENS = 1500

STRUCTURE_TYPES = {
    "clustered": "K=3~6 个高斯簇，簇内密集、簇间分离（可加少量扰动）。最近邻贪心易在簇间反复跳转，NN2opt 容易劣化；2-opt 类全局搜索表现更好",
    "grid": "规则网格（如 5x9 / 6x8 等），点近似等距、无明显长边。最近邻构造效果好，NN2opt 通常占优",
    "ring": "环形/圆形分布（点沿圆周分布，可加少量径向/角度扰动）。最近邻沿圆周构造自然，NN2opt 表现好",
    "corridor": "窄带走廊（点沿一条弯曲或斜向窄带分布，带宽远小于带长，近似一维）。考验构造启发式与局部搜索",
    "outliers": "均匀分布 + 少量远距离群点（2~5 个）。离群点制造长跳，破坏最近邻贪心的短边偏好",
    "mixed": "混合结构（如两个簇 + 一条走廊、或簇 + 离群点等），自行设计能拉开算法差距的复合几何",
}

GENERATOR_SYSTEM_PROMPT = """你是一名"算法组合（Algorithm Portfolio）"研究专家，任务是【生成有区分度的 TSP 实例】。

背景：给定 4 个 TSP 求解算法（各 2 秒单 CPU 预算），我们希望生成坐标实例，使 4 个算法在实例上的解质量【明显分化】——即最强算法与最弱算法的回路长度差距尽量大。这类"难实例"用于验证"没有万能算法、需要算法选择（Algorithm Selection / Portfolio）"的论点。

算法池（各 2s 预算，结果取预算内最佳）：
- "GA"    : 遗传算法（顺序交叉 OX + 交换变异，无局部搜索），种群 200。无局部搜索，在难几何结构上收敛质量明显差。
- "GA2opt": 遗传算法 + 2-opt 局部搜索。探索 + 精修，最稳健，多数几何下接近最优。
- "NN2opt": 多起点最近邻建路 + 2-opt。速度极快；但在聚类/走廊/离群点等几何下，最近邻贪心易被误导，解质量下降。
- "RI2opt": 多次重启随机插入 + 2-opt。通用性好；在强聚类/窄走廊等结构下可能略逊于定向构造。

你被指定生成结构类型：{structure}。
要求：
1. 生成恰好 n 个城市（n 在 40~60 之间自由选择整数），坐标为 [0,1]² 内的浮点数；务必确保 coords 数组长度与 n 完全一致（先定 n，再逐个列出坐标并计数）。
2. 几何结构要真实（避免所有点排成一条完美直线、大量重复点等退化；但"窄走廊"本就是近似一维，允许带宽很小）。
3. 让算法性能分化明显：设计几何使预期强算法与预期弱算法差距尽量大。
4. 坐标带 2~3 位小数即可，保证点在平面内分布。

只输出一个 JSON 对象（不要输出任何其它文字，不要用 Markdown 代码块包裹）：
{{"structure": "{structure}", "n": <40-60的整数>, "coords": [[x,y],...共n个], "expected_strong": ["<1-2个算法>"], "expected_weak": ["<1-2个算法>"], "reasoning": "1-3句话：说明该几何如何放大算法差距"}}"""


def _build_user_prompt(structure):
    return (f"请生成结构类型为 {structure} 的有区分度 TSP 实例。\n"
            f"结构说明：{STRUCTURE_TYPES[structure]}\n"
            f"请按系统提示要求输出 JSON。")


def validate_coords(obj, structure):
    """校验 LLM 输出的实例对象，返回 (n, coords) 或 None。

    注：LLM 常出现声明 n 与实际坐标数不符（计数不准），故以 len(coords) 为准，
    只要实际点数落在 40~60 区间即接受。
    """
    try:
        coords = obj.get("coords")
        if not isinstance(coords, list):
            return None
        n = len(coords)  # 以实际坐标数为准
        if not (40 <= n <= 60):
            return None
        pts = []
        for c in coords:
            if not isinstance(c, (list, tuple)) or len(c) != 2:
                return None
            x, y = float(c[0]), float(c[1])
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                return None
            pts.append((round(x, 4), round(y, 4)))
        # 去重检查：过少唯一点视为退化
        if len(set(pts)) < n * 0.6:
            return None
        return n, pts
    except (TypeError, ValueError):
        return None


def generate_one(chat, structure, seed, max_attempts=3):
    """生成单个有区分度实例。成功返回 dict，最终失败返回 None。"""
    rng = random.Random(seed)
    for _ in range(max_attempts):
        try:
            content, usage = chat.complete([
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT.format(structure=structure)},
                {"role": "user", "content": _build_user_prompt(structure)},
            ], max_tokens=GENERATOR_MAX_TOKENS)
        except Exception as e:
            return {"structure": structure, "ok": False, "error": str(e), "usage": None}
        obj = _extract_json(content)
        if obj is None:
            continue
        res = validate_coords(obj, structure)
        if res is None:
            continue
        n, coords = res
        return {
            "structure": structure,
            "n": n,
            "coords": coords,
            "expected_strong": obj.get("expected_strong", []),
            "expected_weak": obj.get("expected_weak", []),
            "reasoning": str(obj.get("reasoning", "")),
            "ok": True,
            "usage": usage,
        }
    return {"structure": structure, "ok": False, "error": "生成重试耗尽", "usage": None}


_thread_local = threading.local()


def get_thread_chat():
    if not hasattr(_thread_local, "chat"):
        _thread_local.chat = ZhipuChat(max_retries=5)
    return _thread_local.chat