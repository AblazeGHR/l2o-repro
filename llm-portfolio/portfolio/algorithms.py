# -*- coding: utf-8 -*-
"""四个可移植 TSP 求解算法（时间预算制）。

算法池（每个算法单实例 2 秒 CPU 预算）:
    1. GA      —— 遗传算法（顺序交叉 OX + 交换变异，无局部搜索）
    2. GA2opt  —— 遗传算法 + 对部分个体施加 2-opt 局部搜索
    3. NN2opt  —— 最近邻启发式建初始回路 + 2-opt（多起点）
    4. RI2opt  —— 随机插入启发式建初始回路 + 2-opt（多次重启）

所有算法接受 (cities, dist, budget, rng)，返回 (length, tour, info)。
参考移植自 ga-tsp-visualizer/tsp_ga.py，算子改为显式 rng 以便实例级复现。
"""

import math
import random
import time

from .instances import distance_matrix, route_cost


# ---------------- 遗传算子（显式 rng 版本） ----------------

def tournament_select(pop, costs, rng, k=3):
    best = None
    for _ in range(k):
        idx = rng.randrange(len(pop))
        if best is None or costs[idx] < costs[best]:
            best = idx
    return list(pop[best])


def order_crossover(p1, p2, rng):
    """顺序交叉 OX：保留 p1 中一段连续子路径，其余按 p2 顺序补全。"""
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


def swap_mutation(route, rng, rate=0.1):
    n = len(route)
    r = list(route)
    for i in range(n):
        if rng.random() < rate:
            j = rng.randrange(n)
            r[i], r[j] = r[j], r[i]
    return r


def two_opt(tour, dist, max_passes=None):
    """2-opt 局部搜索（best-improvement 扫描，直到不再改进或达到最大轮数）。"""
    n = len(tour)
    r = list(tour)
    passes = 0
    improved = True
    while improved:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                i1 = (i + 1) % n
                j1 = (j + 1) % n
                if i1 == j:
                    continue
                old = dist[r[i]][r[i1]] + dist[r[j]][r[j1]]
                new = dist[r[i]][r[j]] + dist[r[i1]][r[j1]]
                if new < old - 1e-12:
                    r[i + 1:j + 1] = reversed(r[i + 1:j + 1])
                    improved = True
        if max_passes and passes >= max_passes:
            break
    return r


# ---------------- 构造式启发 ----------------

def nearest_neighbor_tour(dist, start=0):
    """最近邻贪心回路，返回 (tour, length)。"""
    n = len(dist)
    unvisited = set(range(n))
    cur = start
    unvisited.remove(cur)
    tour = [cur]
    length = 0.0
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist[cur][j])
        length += dist[cur][nxt]
        cur = nxt
        unvisited.remove(cur)
        tour.append(cur)
    length += dist[cur][tour[0]]
    return tour, length


def random_insertion_tour(dist, rng):
    """随机插入启发式回路，返回 (tour, length)。"""
    n = len(dist)
    start = rng.randrange(n)
    second = rng.randrange(n - 1)
    if second >= start:
        second += 1
    tour = [start, second]
    remaining = [c for c in range(n) if c != start and c != second]
    while remaining:
        city = remaining.pop(rng.randrange(len(remaining)))
        # 在使回路增量最小的位置插入
        best_pos, best_inc = 0, float("inf")
        m = len(tour)
        for k in range(m):
            a, b = tour[k], tour[(k + 1) % m]
            inc = dist[a][city] + dist[city][b] - dist[a][b]
            if inc < best_inc:
                best_inc, best_pos = inc, k + 1
        tour.insert(best_pos, city)
    return tour, route_cost(tour, dist)


# ---------------- 时间预算版 GA ----------------

def _ga_run(cities, dist, budget, rng, use_2opt=False):
    n = len(cities)
    pop_size = 60 if use_2opt else 200
    elite = 3
    two_opt_prob = 0.5  # GA2opt 中施加 2-opt 的子代比例

    pop = [rng.sample(range(n), n) for _ in range(pop_size)]
    if use_2opt:
        pop = [two_opt(ind, dist) for ind in pop]

    best_len, best_tour = float("inf"), None
    start_t = time.perf_counter()
    gen = 0
    while time.perf_counter() - start_t < budget:
        gen += 1
        costs = [route_cost(ind, dist) for ind in pop]
        bi = min(range(pop_size), key=lambda i: costs[i])
        if costs[bi] < best_len:
            best_len, best_tour = costs[bi], list(pop[bi])

        ranked = sorted(range(pop_size), key=lambda i: costs[i])
        new_pop = [list(pop[i]) for i in ranked[:elite]]

        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, costs, rng)
            p2 = tournament_select(pop, costs, rng)
            child = order_crossover(p1, p2, rng) if rng.random() < 0.8 else list(p1)
            child = swap_mutation(child, rng, rate=0.1)
            if use_2opt and rng.random() < two_opt_prob:
                child = two_opt(child, dist)
            new_pop.append(child)
        pop = new_pop

    return best_len, best_tour, {"generations": gen}


# ---------------- 算法池统一入口 ----------------

def solve_ga(cities, dist, budget, rng):
    return _ga_run(cities, dist, budget, rng, use_2opt=False)


def solve_ga2opt(cities, dist, budget, rng):
    return _ga_run(cities, dist, budget, rng, use_2opt=True)


def solve_nn2opt(cities, dist, budget, rng):
    """多起点最近邻 + 2-opt：每个城市作起点建 NN 回路，2-opt 改进，取最优。"""
    n = len(cities)
    best_len, best_tour = float("inf"), None
    start_t = time.perf_counter()
    for s in range(n):
        if time.perf_counter() - start_t >= budget:
            break
        tour, _ = nearest_neighbor_tour(dist, start=s)
        tour = two_opt(tour, dist)
        length = route_cost(tour, dist)
        if length < best_len:
            best_len, best_tour = length, list(tour)
    return best_len, best_tour, {"restarts": n}


def solve_ri2opt(cities, dist, budget, rng):
    """多次重启随机插入 + 2-opt：时间预算内不断重启，取最优。"""
    best_len, best_tour = float("inf"), None
    start_t = time.perf_counter()
    restarts = 0
    while time.perf_counter() - start_t < budget:
        restarts += 1
        tour, _ = random_insertion_tour(dist, rng)
        tour = two_opt(tour, dist)
        length = route_cost(tour, dist)
        if length < best_len:
            best_len, best_tour = length, list(tour)
    return best_len, best_tour, {"restarts": restarts}


ALGORITHMS = {
    "GA": solve_ga,
    "GA2opt": solve_ga2opt,
    "NN2opt": solve_nn2opt,
    "RI2opt": solve_ri2opt,
}

# 算法名 -> 稳定的种子偏移（避免依赖 hash() 的进程随机化）
ALG_SEED_OFFSET = {name: i * 7919 for i, name in enumerate(ALGORITHMS)}


def run_all_algorithms(cities, dist, budget, seed):
    """对单个实例运行全部 4 个算法，返回 {alg: {"length":..,"time":..,"info":..}}。"""
    results = {}
    for name, fn in ALGORITHMS.items():
        rng = random.Random(seed + ALG_SEED_OFFSET[name])
        t0 = time.perf_counter()
        length, tour, info = fn(cities, dist, budget, rng)
        results[name] = {
            "length": round(length, 9),
            "time": round(time.perf_counter() - t0, 4),
            "info": info,
        }
    return results
