# -*- coding: utf-8 -*-
"""实例数值特征提取（供 LLM 选择器输入）。

特征均为节点数 + 坐标统计 + 快速启发式回路长度估计，
全部数值型，便于 LLM 依据特征推断几何结构与求解难度。
"""

import math
import random

from .instances import distance_matrix
from .algorithms import nearest_neighbor_tour, random_insertion_tour


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def extract_features(cities, feature_seed=12345):
    """返回实例的数值特征 dict。"""
    n = len(cities)
    xs = [c[0] for c in cities]
    ys = [c[1] for c in cities]
    dist = distance_matrix(cities)

    # 最近邻距离分布（row[0]=0 为自身）
    nn = []
    nn2 = []
    for i in range(n):
        row = sorted(dist[i])
        nn.append(row[1])
        nn2.append(row[2])

    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)

    # 快速启发式回路（固定起点/固定种子，保证确定性）
    nn_tour, nn_len = nearest_neighbor_tour(dist, start=0)
    ri_rng = random.Random(feature_seed)
    ri_tour, ri_len = random_insertion_tour(dist, ri_rng)

    exp_spacing = 1.0 / math.sqrt(n)  # 均匀分布下的期望点间距
    mean_nn = _mean(nn)
    std_nn = _std(nn)

    def r(v):
        return round(v, 4)

    return {
        "n": n,
        "mean_x": r(_mean(xs)),
        "std_x": r(_std(xs)),
        "mean_y": r(_mean(ys)),
        "std_y": r(_std(ys)),
        "bbox_width": r(bbox_w),
        "bbox_height": r(bbox_h),
        "bbox_diag": r(math.hypot(bbox_w, bbox_h)),
        "mean_nn_dist": r(mean_nn),
        "min_nn_dist": r(min(nn)),
        "max_nn_dist": r(max(nn)),
        "std_nn_dist": r(std_nn),
        "mean_2nn_dist": r(_mean(nn2)),
        # >1 表示比均匀分布更分散，<1 表示更聚集
        "nn_spread_ratio": r(mean_nn / exp_spacing),
        # 最近邻距离变异系数（几何均匀性的度量）
        "nn_cv": r(std_nn / mean_nn),
        "nn_tour_len": r(nn_len),
        "ri_tour_len": r(ri_len),
        "nn_over_ri": r(nn_len / ri_len),
        "quick_tour_per_node": r((nn_len + ri_len) / 2 / n),
    }
