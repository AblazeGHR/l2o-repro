# -*- coding: utf-8 -*-
"""LLM 调用冒烟测试：验证 key 读取 / 白名单 / thinking 关闭 / JSON 输出 / usage。

用法：ZHIPU_API_KEY=xxx python src/smoke_test.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluator import make_instance  # noqa: E402
from llm_advisor import ALLOWED_MODELS, MODEL, extract_json, suggest  # noqa: E402


def main():
    assert os.environ.get("ZHIPU_API_KEY"), "ZHIPU_API_KEY 未设置"
    assert MODEL in ALLOWED_MODELS, f"白名单校验失败: {MODEL}"

    # 用空历史 + 前 1 轮做一次真实调用
    hist = []
    for i in range(1, 9):
        h = {
            "population": 40 + i * 15, "crossover_rate": 0.5 + i * 0.05,
            "mutation_rate": 0.1 + i * 0.04, "generations": 120 + i * 30,
            "best_cost": round(5.0 - i * 0.1, 4),
        }
        hist.append(h)

    sugg, raw, usage, fallback = suggest(hist, instance_idx=0, round_idx=0)
    print("MODEL:", MODEL)
    print("WHITELIST:", sorted(ALLOWED_MODELS))
    print("FALLBACK:", fallback)
    print("USAGE:", usage)
    print("N_SUGGESTIONS:", len(sugg))
    for s in sugg[:3]:
        print("  ", s)
    # 校验范围
    for s in sugg:
        assert 20 <= s["population"] <= 200
        assert 0.5 <= s["crossover_rate"] <= 1.0
        assert 0.05 <= s["mutation_rate"] <= 0.5
        assert 100 <= s["generations"] <= 500
        assert s.get("reason")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
