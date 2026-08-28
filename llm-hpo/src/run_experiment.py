# -*- coding: utf-8 -*-
"""LLM-HPO vs Optuna TPE vs Random Search 实验编排脚本。

三个阶段（每个实例有独立 checkpoint，中断后重启可续跑）：
  1) LLM-HPO：每轮把"已试超参 + GA 最优成本"表格交给 LLM，in-context 建议 8 组；
  2) TPE：Optuna TPE 采样，同评估预算；
  3) Random Search：均匀随机采样，同评估预算。

全部评估都真实调用 GA；GA 种子 = hash(实例, 超参)，保证确定性。
用法：HPO_WORKERS=6 CPU_WORKERS=8 python src/run_experiment.py
"""
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluator import (BATCH, N_EVALS, N_INSTANCES, N_ROUNDS, SPACE,  # noqa: E402
                       evaluate, make_instance)
from llm_advisor import suggest  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
CKPT = os.path.join(RESULTS, "checkpoints")

DEFAULT_LLM_WORKERS = 6
DEFAULT_CPU_WORKERS = min(16, os.cpu_count() or 4)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def inst_path(method, i):
    return os.path.join(CKPT, f"{method}_inst_{i:02d}.jsonl")


def read_ckpt(method, i):
    p = inst_path(method, i)
    rows = []
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def append_ckpt(method, i, row):
    p = inst_path(method, i)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------- 阶段 1：LLM-HPO ----------------

def run_llm_inst(i):
    """单个实例的 LLM-HPO 循环（轮次串行，checkpoint 续跑）。"""
    history = []
    rows = read_ckpt("llm_hpo", i)
    start_round = len(rows)
    for row in rows:
        for s in row["suggestions"]:
            history.append(s)

    if start_round >= N_ROUNDS:
        log(f"[LLM-HPO] 实例 {i:02d} 已完成（{start_round} 轮），跳过")
        return i, "ok"

    for r in range(start_round, N_ROUNDS):
        # 单轮重试：LLM/网络偶发错误时等待后重试，最多 3 次
        out_sugs, raw, usage, fallback, best_in_round = None, "", {"prompt_tokens": 0, "completion_tokens": 0}, False, None
        for attempt in range(3):
            try:
                suggestions, raw, usage, fallback = suggest(history, i, r)
                out_sugs = []
                for s in suggestions:
                    cost = evaluate(i, s)
                    s2 = dict(s)
                    s2["best_cost"] = cost
                    out_sugs.append(s2)
                    history.append(s2)
                best_in_round = min(x["best_cost"] for x in out_sugs)
                break
            except Exception as e:
                if attempt >= 2:
                    raise
                log(f"[LLM-HPO] 实例 {i:02d} 第 {r + 1} 轮出错(第{attempt + 1}次): {e!r}，{5 * (attempt + 1)}s 后重试")
                time.sleep(5 * (attempt + 1))
        if out_sugs is None:
            raise RuntimeError(f"实例 {i} 第 {r} 轮连续 3 次失败")

        row = {
            "instance": i,
            "round": r,
            "suggestions": out_sugs,
            "usage": usage,
            "fallback": fallback,
            "raw": (raw or "")[:3000],
        }
        append_ckpt("llm_hpo", i, row)

        best_so_far = min(x["best_cost"] for x in history)
        log(f"[LLM-HPO] 实例 {i:02d} 第 {r + 1}/{N_ROUNDS} 轮完成 | "
            f"本轮最好 {best_in_round:.4f} | "
            f"累计最好 {best_so_far:.4f} | tokens {usage['prompt_tokens']}+{usage['completion_tokens']}"
            + (" | [fallback]" if fallback else ""))
    return i, "ok"


# ---------------- 阶段 2：Optuna TPE ----------------

def run_tpe_inst(i):
    import optuna  # 延迟导入，worker 进程各自 import

    rows = read_ckpt("tpe", i)
    done = len(rows)
    if done >= N_EVALS:
        log(f"[TPE] 实例 {i:02d} 已完成（{done} 次评估），跳过")
        return i, "ok"

    sampler = optuna.samplers.TPESampler(seed=5000 + i)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    param_store = {}

    def objective(trial):
        h = {
            "population": trial.suggest_int("population", *SPACE["population"]),
            "crossover_rate": round(trial.suggest_float("crossover_rate", *SPACE["crossover_rate"]), 3),
            "mutation_rate": round(trial.suggest_float("mutation_rate", *SPACE["mutation_rate"]), 3),
            "generations": trial.suggest_int("generations", *SPACE["generations"]),
        }
        param_store[trial.number] = h
        return evaluate(i, h)

    for n in range(done, N_EVALS):
        trial = study.ask()
        cost = objective(trial)
        study.tell(trial, cost)
        h = param_store[trial.number]
        append_ckpt("tpe", i, {"trial": n, "h": h, "cost": cost})
        if (n + 1) % 20 == 0:
            rows = read_ckpt("tpe", i)
            best = min(r["cost"] for r in rows)
            log(f"[TPE] 实例 {i:02d} 评估 {n + 1}/{N_EVALS} | 累计最好 {best:.4f}")
    return i, "ok"


# ---------------- 阶段 3：Random Search ----------------

def run_random_inst(i):
    rows = read_ckpt("random", i)
    done = len(rows)
    if done >= N_EVALS:
        log(f"[RAND] 实例 {i:02d} 已完成（{done} 次评估），跳过")
        return i, "ok"

    rng = random.Random(7000 + i)
    for n in range(done, N_EVALS):
        h = {
            "population": rng.randint(*SPACE["population"]),
            "crossover_rate": round(rng.uniform(*SPACE["crossover_rate"]), 3),
            "mutation_rate": round(rng.uniform(*SPACE["mutation_rate"]), 3),
            "generations": rng.randint(*SPACE["generations"]),
        }
        cost = evaluate(i, h)
        append_ckpt("random", i, {"trial": n, "h": h, "cost": cost})
        if (n + 1) % 20 == 0:
            rows = read_ckpt("random", i)
            best = min(r["cost"] for r in rows)
            log(f"[RAND] 实例 {i:02d} 评估 {n + 1}/{N_EVALS} | 累计最好 {best:.4f}")
    return i, "ok"


# ---------------- 主流程 ----------------

def run_phase_threads(name, func, workers):
    """LLM-HPO 阶段用线程并发（网络 IO 密集，避免 Windows spawn 下的原生崩溃）。"""
    log(f"===== 阶段 {name} 开始（{N_INSTANCES} 实例，线程并发 {workers}）=====")
    t0 = time.time()
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(func, i): i for i in range(N_INSTANCES)}
        for f in as_completed(futs):
            i = futs[f]
            try:
                idx, status = f.result()
                if status != "ok":
                    failed.append(idx)
            except Exception as e:
                failed.append(i)
                log(f"[ERROR] 实例 {i:02d} 失败: {e!r}")
    log(f"===== 阶段 {name} 结束，耗时 {(time.time() - t0) / 60:.1f} 分钟，失败 {len(failed)}: {failed} =====")
    return failed


def run_phase(name, func, workers):
    log(f"===== 阶段 {name} 开始（{N_INSTANCES} 实例，{workers} 并发）=====")
    t0 = time.time()
    failed = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(func, i): i for i in range(N_INSTANCES)}
        for f in as_completed(futs):
            i = futs[f]
            try:
                idx, status = f.result()
                if status != "ok":
                    failed.append(idx)
            except Exception as e:
                failed.append(i)
                log(f"[ERROR] 实例 {i:02d} 失败: {e!r}")
    log(f"===== 阶段 {name} 结束，耗时 {(time.time() - t0) / 60:.1f} 分钟，失败 {len(failed)}: {failed} =====")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-workers", type=int, default=int(os.environ.get("HPO_WORKERS", DEFAULT_LLM_WORKERS)))
    ap.add_argument("--cpu-workers", type=int, default=int(os.environ.get("CPU_WORKERS", DEFAULT_CPU_WORKERS)))
    args = ap.parse_args()

    os.makedirs(CKPT, exist_ok=True)
    with open(os.path.join(RESULTS, "PID"), "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    log(f"PID={os.getpid()} | 实验启动 | 实例数={N_INSTANCES} 轮数={N_ROUNDS} 每轮批次={BATCH} "
        f"评估预算/实例/方法={N_EVALS} | llm_workers={args.llm_workers} cpu_workers={args.cpu_workers}")
    log(f"模型白名单校验: 仅允许 {sorted({'glm-4.5-air'})}")
    log(f"ZHIPU_API_KEY 已设置: {bool(os.environ.get('ZHIPU_API_KEY'))}")

    # 记录实例信息（可复现）
    with open(os.path.join(RESULTS, "instances.json"), "w", encoding="utf-8") as f:
        json.dump({"tsp_n": 20, "n_instances": N_INSTANCES,
                   "coords": [make_instance(i) for i in range(N_INSTANCES)]}, f, ensure_ascii=False)

    all_ok = True
    # LLM-HPO：线程并发；TPE/Random：进程并发（纯计算，无网络）
    phases = [("LLM-HPO", run_llm_inst, args.llm_workers, run_phase_threads),
              ("TPE", run_tpe_inst, args.cpu_workers, run_phase),
              ("RANDOM", run_random_inst, args.cpu_workers, run_phase)]
    for name, func, w, runner in phases:
        failed = runner(name, func, w)
        if failed:
            all_ok = False
            log(f"[严重] 阶段 {name} 有实例失败: {failed}；请检查后重启续跑")

    if all_ok:
        with open(os.path.join(RESULTS, "DONE"), "w", encoding="utf-8") as f:
            f.write(f"DONE at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log("全部阶段完成，已写入 results/DONE")
    else:
        log("存在失败实例，未写 DONE；重启可续跑未完成部分")
        sys.exit(1)


if __name__ == "__main__":
    main()
