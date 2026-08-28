#!/usr/bin/env python
"""Evaluate a trained AM model on a fixed TSP20 test set and compare with
classic baselines (nearest neighbour, 2-opt, random insertion, OR-Tools).

Usage:
    python eval_trained.py <model_dir> [--test_size 1000] [--seed 1234]
"""
import argparse
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import load_model, move_to
from nets.attention_model import set_decode_type


def eval_model_greedy(model, dataset, batch_size, device):
    set_decode_type(model, "greedy")
    model.eval()
    model.to(device)
    costs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size):
            batch = move_to(batch, device)
            cost, _ = model(batch)
            costs.append(cost.cpu())
    return torch.cat(costs).numpy()


# ---------------- classical heuristics (numpy, no torch) ----------------

def tour_length(tour, pts):
    pts = pts[tour]
    return np.sum(np.sqrt(((np.roll(pts, -1, axis=0) - pts) ** 2).sum(axis=1)))


def nearest_neighbor(pts):
    n = len(pts)
    tour = [0]
    remaining = set(range(1, n))
    cur = 0
    while remaining:
        nxt = min(remaining, key=lambda j: np.sum((pts[cur] - pts[j]) ** 2))
        tour.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return np.array(tour)


def two_opt(pts, max_iter=200):
    tour = nearest_neighbor(pts)
    n = len(tour)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # delta from reversing segment (i..j)
                a, b, c, d = tour[i - 1], tour[i], tour[j], tour[(j + 1) % n]
                d0 = dist(pts[a], pts[b]) + dist(pts[c], pts[d])
                d1 = dist(pts[a], pts[c]) + dist(pts[b], pts[d])
                if d1 + 1e-12 < d0:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour


def dist(p, q):
    return math.sqrt(((p - q) ** 2).sum())


def random_insertion(pts, rng):
    n = len(pts)
    order = rng.permutation(n)
    tour = [order[0], order[1]]
    for idx in order[2:]:
        p = pts[idx]
        best_pos, best_delta = 0, float("inf")
        for k in range(len(tour)):
            a, b = pts[tour[k]], pts[tour[(k + 1) % len(tour)]]
            delta = dist(a, p) + dist(p, b) - dist(a, b)
            if delta < best_delta:
                best_delta, best_pos = delta, k + 1
        tour.insert(best_pos, idx)
    return np.array(tour)


def ortools_solve(pts, time_limit_s=0.5):
    try:
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    except ImportError:
        return None, None
    n = len(pts)
    dmat = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    dmat = (dmat * 10000).astype(int)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(i, j):
        return int(dmat[manager.IndexToNode(i), manager.IndexToNode(j)])

    dist_cb_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_idx)
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.FromMilliseconds(int(time_limit_s * 1000))

    sol = routing.SolveWithParameters(search_params)
    if sol is None:
        return None, None
    idx = routing.Start(0)
    tour = []
    while not routing.IsEnd(idx):
        tour.append(manager.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    tour = np.array(tour)
    length = tour_length(tour, pts)
    return tour, length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--epoch", type=int, default=None,
                    help="Checkpoint epoch to load (default: max epoch found in model_dir). "
                         "Explicitly specify e.g. 79 to avoid accidentally loading an untrained epoch-0 model.")
    ap.add_argument("--test_size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--no_cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda:0")
    model, _ = load_model(args.model_dir, epoch=args.epoch)
    model.to(device)
    print(f"loaded model from {args.model_dir}"
          + (f" (epoch {args.epoch})" if args.epoch is not None else " (max epoch)"))

    # fixed test set
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    dataset = model.problem.make_dataset(size=20, num_samples=args.test_size)

    pts = np.array([d.numpy() for d in dataset.data])  # shape (N, graph_size, 2)

    t0 = time.time()
    model_costs = eval_model_greedy(model, dataset, args.batch_size, device)
    model_time = time.time() - t0

    nn_costs, opt2_costs, ri_costs, ort_costs = [], [], [], []
    n_ort = 0
    t0 = time.time()
    for i in range(len(pts)):
        p = pts[i]
        t_nn = nearest_neighbor(p)
        nn_costs.append(tour_length(t_nn, p))
        opt2_costs.append(tour_length(two_opt(p), p))
        ri_costs.append(tour_length(random_insertion(p, rng), p))
        t_ort, c_ort = ortools_solve(p, time_limit_s=0.5)
        if t_ort is not None:
            n_ort += 1
            ort_costs.append(c_ort)
    heur_time = time.time() - t0

    rows = [
        ("Attention Model (greedy, ours)", model_costs.mean(), model_costs.std()),
        ("Nearest Neighbour", np.mean(nn_costs), np.std(nn_costs)),
        ("2-opt (NN init)", np.mean(opt2_costs), np.std(opt2_costs)),
        ("Random Insertion", np.mean(ri_costs), np.std(ri_costs)),
    ]
    if n_ort:
        rows.append((f"OR-Tools GLS (n={n_ort})", np.mean(ort_costs), np.std(ort_costs)))

    print(f"\n=== TSP20 test set: {len(pts)} instances, seed {args.seed} ===")
    print(f"model greedy took {model_time:.1f}s, heuristics took {heur_time:.1f}s")
    print(f"{'method':<32}{'mean tour':>10}{'std':>8}")
    for name, mean, std in rows:
        print(f"{name:<32}{mean:>10.4f}{std:>8.4f}")

    # save results
    import os
    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", "comparison_tsp20.csv")
    with open(out, "w") as f:
        f.write("method,mean_tour_length,std\n")
        for name, mean, std in rows:
            f.write(f"{name},{mean:.4f},{std:.4f}\n")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
