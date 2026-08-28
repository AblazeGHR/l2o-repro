# -*- coding: utf-8 -*-
"""共享的搜索空间 / 实例 / 确定性评估函数。

关键：GA 的随机种子 = hash(实例下标, 超参)，因此"同一(实例, 超参) → 同一 GA 结果"，
LLM-HPO / TPE / Random 三方法对相同超参的评估完全一致，对比公平、可复现。
"""
import hashlib

from tsp_ga import make_random_instance, solve_tsp

# ---- 实验规模 ----
TSP_N = 20            # 城市数
N_INSTANCES = 30      # TSP20 实例数
N_ROUNDS = 30         # LLM-HPO 轮数
BATCH = 8             # 每轮 LLM 建议的超参组数
N_EVALS = N_ROUNDS * BATCH  # 240：每实例每方法的评估预算

# ---- 超参数搜索空间 ----
SPACE = {
    "population": (20, 200),        # 整数
    "crossover_rate": (0.5, 1.0),   # 浮点
    "mutation_rate": (0.05, 0.5),   # 浮点
    "generations": (100, 500),      # 整数
}


def make_instance(idx):
    """第 idx 个 TSP20 实例（确定性生成）。"""
    return make_random_instance(TSP_N, seed=1000 + idx)


def ga_seed(instance_idx, h):
    """由 (实例, 超参) 派生确定性 GA 随机种子。"""
    s = f"{instance_idx}|{int(h['population'])}|{float(h['crossover_rate']):.6f}|" \
        f"{float(h['mutation_rate']):.6f}|{int(h['generations'])}"
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def evaluate(instance_idx, h):
    """对实例 instance_idx 用超参 h 真实跑一次 GA，返回最优路径成本（越小越好）。"""
    cities = make_instance(instance_idx)
    best, _, _ = solve_tsp(
        cities,
        seed=ga_seed(instance_idx, h),
        pop_size=int(h["population"]),
        generations=int(h["generations"]),
        crossover_rate=float(h["crossover_rate"]),
        mutation_rate=float(h["mutation_rate"]),
    )
    return best


def random_suggestion(rng):
    """在搜索空间内随机采样一组超参。"""
    return {
        "population": rng.randint(*SPACE["population"]),
        "crossover_rate": round(rng.uniform(*SPACE["crossover_rate"]), 3),
        "mutation_rate": round(rng.uniform(*SPACE["mutation_rate"]), 3),
        "generations": rng.randint(*SPACE["generations"]),
    }
