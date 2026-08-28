# -*- coding: utf-8 -*-
"""TSP 实例生成与路径评估工具。"""

import math
import random


def make_random_instance(n, rng):
    """在 [0,1]^2 单位正方形内生成 n 个城市的坐标。"""
    return [(rng.random(), rng.random()) for _ in range(n)]


def distance_matrix(cities):
    """由城市坐标计算两两欧氏距离矩阵（对称）。"""
    n = len(cities)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = cities[i]
        for j in range(i + 1, n):
            d = math.hypot(xi - cities[j][0], yi - cities[j][1])
            dist[i][j] = dist[j][i] = d
    return dist


def route_cost(route, dist):
    """计算一条回路的总长度（首尾相连）。"""
    n = len(route)
    return sum(dist[route[i]][route[(i + 1) % n]] for i in range(n))


def generate_instances(n_instances, seed, sizes=(15, 20, 25, 30)):
    """生成 n_instances 个 TSP 实例，规模在 sizes 间轮转混合。

    返回 [(idx, n, cities, seed)]，seed 用于实例级复现。
    """
    rng = random.Random(seed)
    instances = []
    for i in range(n_instances):
        n = sizes[i % len(sizes)]
        inst_seed = seed + i * 1000 + 1
        local_rng = random.Random(inst_seed)
        cities = make_random_instance(n, local_rng)
        instances.append((i, n, cities, inst_seed))
    return instances
