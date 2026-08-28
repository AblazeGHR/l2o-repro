# -*- coding: utf-8 -*-
"""GA 解 TSP20 —— 移植自 ga-tsp-visualizer/tsp_ga.py，保持遗传算子语义不变。

算子：锦标赛选择（k=3）/ 顺序交叉 OX / 交换变异（逐位置概率）/ 精英保留。
差异：fitness 用 numpy 向量化；随机源统一为传入的 random.Random(seed)，
使"同一 (实例, 超参) → 同一 GA 结果"，保证三方法公平可复现。
"""
import random

import numpy as np


def make_random_instance(n, seed=None):
    """在 [0,1]^2 平面内随机生成 n 个城市的坐标 (x, y)。"""
    rng = random.Random(seed)
    return [(rng.random(), rng.random()) for _ in range(n)]


def distance_matrix(cities):
    """由城市坐标计算两两欧氏距离矩阵（对称）。"""
    n = len(cities)
    c = np.asarray(cities, dtype=float)
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    return d


def route_costs(pop, dist):
    """pop: list[list[int]] 种群；返回 numpy 数组形式的各回路总长（首尾相连）。"""
    P = np.asarray(pop, dtype=np.int64)
    seg = dist[P[:, :-1], P[:, 1:]]
    close = dist[P[:, -1], P[:, 0]]
    return seg.sum(axis=1) + close


def tournament_select(pop, costs, rng, k=3):
    """锦标赛选择：随机抽 k 个个体，返回其中总路程最短的一个。"""
    n = len(pop)
    best = None
    for _ in range(k):
        idx = rng.randrange(n)
        if best is None or costs[idx] < costs[best]:
            best = idx
    return list(pop[best])


def order_crossover(p1, p2, rng):
    """顺序交叉 OX：保留 p1 中一段连续子路径，其余城市按 p2 中的顺序补全。"""
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[a:b] = p1[a:b]
    keep = set(child[a:b])
    fill = [g for g in p2 if g not in keep]
    j = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[j]
            j += 1
    return child


def swap_mutation(route, rate, rng):
    """交换变异：对每个位置，以 rate 概率与随机位置交换基因。"""
    n = len(route)
    r = list(route)
    for i in range(n):
        if rng.random() < rate:
            j = rng.randrange(n)
            r[i], r[j] = r[j], r[i]
    return r


def solve_tsp(cities, seed, pop_size=100, generations=300, elite=2,
              tournament_k=3, crossover_rate=0.8, mutation_rate=0.1):
    """遗传算法解 TSP 主函数（cities 由调用方给定，seed 决定整个进化过程）。

    返回: (best_cost, best_route, history)  —— best_cost 为最优回路长度。
    """
    rng = random.Random(seed)
    dist = distance_matrix(cities)
    n_nodes = len(cities)

    pop = [rng.sample(range(n_nodes), n_nodes) for _ in range(pop_size)]
    best_cost, best_route = float("inf"), None
    history = []

    for _gen in range(generations):
        costs = route_costs(pop, dist)
        ranked = sorted(range(pop_size), key=lambda i: costs[i])

        gen_best_cost = float(costs[ranked[0]])
        history.append(gen_best_cost)
        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_route = list(pop[ranked[0]])

        # 精英保留：最优的 elite 个个体直接进入下一代
        new_pop = [list(pop[i]) for i in ranked[:elite]]

        # 选择 + 交叉 + 变异补充满下一代
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, costs, rng, tournament_k)
            p2 = tournament_select(pop, costs, rng, tournament_k)
            if rng.random() < crossover_rate:
                child = order_crossover(p1, p2, rng)
            else:
                child = list(p1)
            child = swap_mutation(child, mutation_rate, rng)
            new_pop.append(child)

        pop = new_pop

    return best_cost, best_route, history
