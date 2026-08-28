# -*- coding: utf-8 -*-
"""OPRO（Optimization by Prompting, Yang et al. ICLR 2024）TSP20 复现——与 LMEA 范式对比。

范式说明：
  - LMEA（LLM 当进化算子）：LLM 扮演交叉/变异算子，嵌在遗传循环里，种群驱动。
  - OPRO（LLM 直接迭代改进解）：维护解池，每轮让 LLM 根据"解→分数"历史直接提出新解，
    并周期性让 LLM 优化提示词本身（meta-prompt）。

流程：20 个 TSP20 实例（seed=1..20，实例生成方式与 lmea-repro/lmea_full.py 完全一致），
每实例 50 轮；解池初始 3 个随机解，保留 top-k=5；每 10 轮做一次提示词优化调用。
记录每轮 best cost / 有效解 / usage，写 CSV；实例结束写 JSON。
全部实例完成后生成对比图、汇总 CSV、README_OPRO.md，写 DONE 标记。

模型白名单硬校验（仅 glm-4.5-air）；key 仅走环境变量（ZHIPU_KEYS 逗号分隔或 K6/K7/K8），
严禁落盘；调用间 1-1.5s 节流；429 指数退避 1→2→4→8s（封顶 8s）最多 6 次。
"""
import csv
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openai import OpenAI

ALLOWED_MODELS = {"glm-4.5-air"}
MODEL = "glm-4.5-air"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
NODE_COUNT = 20
ROUNDS = 50
POOL_K = 5
INIT_POOL = 3
META_EVERY = 10
HIST_N = 5
MAX_TOKENS = 512
MAX_RETRY_429 = 6
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LMEA_DIR = r"D:/notes/Ablaze/pages/理工/计算机/申请导师快速练习项目/lmea-repro/results/full"

THROTTLE = 1.25          # 相邻调用最小间隔（秒），落在 1-1.5s 区间
BACKOFF_CAP = 8.0        # 429 退避封顶

DEFAULT_META_PROMPT = (
    "You are a cooperative agent helping to solve a Traveling Salesman Problem (TSP). "
    "You will be given the coordinates of cities and some example tours with their lengths. "
    "Propose a NEW tour that is SHORTER than all the example tours. "
    "Think step by step: look at the coordinates, note which cities are close to each other, "
    "then construct a tour visiting nearby cities consecutively. "
    "Output exactly one tour in the format <trace>c1,c2,...,c20</trace> "
    "where c1..c20 is a permutation of the integers 0..19. Output nothing else after the trace."
)


def fail(msg):
    print(f"[FATAL] {msg}", flush=True)
    sys.exit(1)


def load_keys():
    """key 只从环境变量读取，严禁写盘。"""
    raw = os.environ.get("ZHIPU_KEYS")
    if raw:
        keys = [k.strip() for k in raw.split(",") if k.strip()]
    else:
        keys = [os.environ.get(f"K{i}") for i in (6, 7, 8)]
        keys = [k for k in keys if k]
    if not keys:
        fail("环境变量未提供 API key（ZHIPU_KEYS 或 K6/K7/K8）")
    return keys


# ---------------------------------------------------------------- 实例与成本（与 LMEA 完全一致）
def gen_tsp(n, seed):
    rng = random.Random(seed)
    return [(round(rng.uniform(0, 100), 2), round(rng.uniform(0, 100), 2)) for _ in range(n)]


def tour_cost(points, tour):
    d = 0.0
    for i in range(len(tour)):
        a = points[tour[i]]
        b = points[tour[(i + 1) % len(tour)]]
        d += math.hypot(a[0] - b[0], a[1] - b[1])
    return round(d, 2)


def points_str(points):
    return ", ".join(f"{i}:({x},{y})" for i, (x, y) in enumerate(points))


def repair_tour(t, n):
    seen, out = set(), []
    for x in t:
        if isinstance(x, int) and 0 <= x < n and x not in seen:
            seen.add(x)
            out.append(x)
    missing = [x for x in range(n) if x not in seen]
    random.shuffle(missing)
    return out + missing


def parse_tours(text, n):
    sols = []
    for m in re.finditer(r"<trace>(.*?)</trace>", text, re.DOTALL):
        raw = m.group(1).replace(" ", "").split(",")
        nums = [int(x) for x in raw if x.strip().lstrip("-").isdigit()]
        if not nums:
            continue
        t = repair_tour(nums, n)
        if sorted(t) != list(range(n)):
            t = repair_tour(t, n)
        if t not in sols:
            sols.append(t)
    return sols


def extract_usage(usage):
    def _g(o, name):
        return getattr(o, name, None)

    prompt = completion = reasoning = total = cached = None
    if usage is not None:
        prompt = _g(usage, "prompt_tokens")
        completion = _g(usage, "completion_tokens")
        total = _g(usage, "total_tokens")
        reasoning = _g(usage, "reasoning_tokens")
        det = _g(usage, "completion_tokens_details")
        if reasoning is None and det is not None:
            reasoning = _g(det, "reasoning_tokens")
        pd = _g(usage, "prompt_tokens_details")
        if pd is not None:
            cached = _g(pd, "cached_tokens")
    return {
        "prompt_tokens": prompt or 0,
        "completion_tokens": completion or 0,
        "reasoning_tokens": reasoning,
        "cached_tokens": cached or 0,
        "total_tokens": total or 0,
    }


# ---------------------------------------------------------------- API 客户端：3 key 轮询 + 节流 + 429 退避
class LLMClient:
    def __init__(self, keys, model):
        if model not in ALLOWED_MODELS:
            fail(f"模型 {model} 不在白名单 {ALLOWED_MODELS}")
        self.model = model
        self.clients = [
            OpenAI(api_key=k, base_url=BASE_URL) for k in keys
        ]
        self.idx = 0
        self.last_call = 0.0

    def _throttle(self):
        wait = THROTTLE - (time.time() - self.last_call)
        if wait > 0:
            time.sleep(wait)
        self.last_call = time.time()

    def _next_client(self):
        c = self.clients[self.idx % len(self.clients)]
        self.idx += 1
        return c

    def chat(self, system, user):
        """单次调用：轮询 key，429 指数退避 1→2→4→8s（封顶 8s）最多 6 次。"""
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=MAX_TOKENS,
            temperature=1.0,
            extra_body={"thinking": {"type": "disabled"}},
        )
        last_err = None
        for attempt in range(MAX_RETRY_429):
            self._throttle()
            client = self._next_client()
            try:
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or "", extract_usage(resp.usage), None
            except Exception as e:
                last_err = e
                is_429 = "429" in str(e) or (getattr(e, "status_code", None) == 429)
                delay = min(BACKOFF_CAP, 2.0 ** attempt) if is_429 else 3.0
                print(f"  [WARN] 调用失败 attempt{attempt + 1} (429={is_429}): {e}，{delay}s 后重试", flush=True)
                time.sleep(delay)
                if not is_429 and attempt >= 2:
                    break
        return None, None, last_err


# ---------------------------------------------------------------- OPRO 核心流程
def initial_pool(points, seed):
    rng = random.Random(seed)
    pool = []
    for _ in range(INIT_POOL):
        t = list(range(NODE_COUNT))
        rng.shuffle(t)
        pool.append((t, tour_cost(points, t)))
    pool.sort(key=lambda x: x[1])
    return pool


def build_user_prompt(points, pool, history, meta_prompt):
    lines = [meta_prompt.strip(), "", f"Cities (index:(x,y)): {points_str(points)}", ""]
    lines.append("Candidate tours (best first):")
    for t, c in pool:
        lines.append(f"<trace>{','.join(map(str, t))}</trace> length: {c}")
    if history:
        lines.append("")
        lines.append("Recent attempts (tour -> length):")
        for t, c in history[-HIST_N:]:
            lines.append(f"<trace>{','.join(map(str, t))}</trace> -> {c}")
    lines.append("")
    lines.append("Propose a new tour shorter than all the above. New tour:")
    return "\n".join(lines)


def run_instance(client, seed):
    points = gen_tsp(NODE_COUNT, seed)
    pool = initial_pool(points, seed)
    init_best = pool[0][1]
    meta_prompt = DEFAULT_META_PROMPT
    prompt_version = 1
    prompt_log = [(1, 0, init_best)]  # (version, round, best_at_switch)
    history = []  # 最近尝试 (tour, cost)
    rows = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "total_tokens": 0}
    meta_calls = 0
    solution_calls = 0

    for rnd in range(1, ROUNDS + 1):
        user = build_user_prompt(points, pool, history, meta_prompt)
        text, usage, err = client.chat(
            "You are an expert optimization assistant. Always answer with a single <trace>...</trace> tour.",
            user,
        )
        for k in usage_total:
            usage_total[k] += (usage or {}).get(k, 0) or 0

        best_cost = pool[0][1]
        valid = 0
        if text is not None:
            solution_calls += 1
            sols = parse_tours(text, NODE_COUNT)
            if sols:
                valid = 1
                for t in sols:
                    c = tour_cost(points, t)
                    pool.append((t, c))
                    history.append((t, c))
                pool.sort(key=lambda x: x[1])
                pool = pool[:POOL_K]
                best_cost = pool[0][1]

        meta_note = ""
        if rnd % META_EVERY == 0 and rnd < ROUNDS:
            # 提示词自优化：根据历史表现让 LLM 改写 meta-prompt
            score_lines = "\n".join(
                f"  - instruction v{v} used from round {r0}: best length reached {b}"
                for v, r0, b in prompt_log
            )
            recent = ", ".join(str(c) for _, c in history[-10:]) or "none yet"
            meta_user = (
                "You are a prompt optimizer. The instruction below is given to a TSP-solver agent, "
                "which reads city coordinates and example tours, and proposes a new shorter tour.\n\n"
                f"Current instruction:\n\"\"\"{meta_prompt}\"\"\"\n\n"
                f"Performance history (tour lengths; lower is better):\n{score_lines}\n"
                f"Most recent tour lengths: {recent}\n\n"
                "Improve the instruction so the solver proposes SHORTER tours "
                "(e.g. emphasize nearest-neighbor construction, avoiding crossing edges, or "
                "checking that all 20 cities appear exactly once). "
                "Keep it under 150 words. "
                "Output only the new instruction wrapped as <prompt>...</prompt>."
            )
            mtext, musage, merr = client.chat(
                "You are an expert prompt engineer for optimization tasks.",
                meta_user,
            )
            for k in usage_total:
                usage_total[k] += (musage or {}).get(k, 0) or 0
            if mtext is not None:
                meta_calls += 1
                m = re.search(r"<prompt>(.*?)</prompt>", mtext, re.DOTALL)
                if m and m.group(1).strip():
                    meta_prompt = m.group(1).strip()
                    prompt_version += 1
                    prompt_log.append((prompt_version, rnd, best_cost))
                    meta_note = f" meta_prompt->v{prompt_version}"
                else:
                    meta_note = " meta_prompt:未解析到<prompt>，保留旧版"
            else:
                meta_note = f" meta_prompt调用失败:{merr}"

        rows.append({
            "round": rnd,
            "best_cost": best_cost,
            "valid": valid,
            "prompt_version": prompt_version,
            "prompt_tokens": (usage or {}).get("prompt_tokens", 0) or 0,
            "completion_tokens": (usage or {}).get("completion_tokens", 0) or 0,
            "cached_tokens": (usage or {}).get("cached_tokens", 0) or 0,
            "total_tokens": (usage or {}).get("total_tokens", 0) or 0,
        })
        print(f"[glm-4.5-air] seed={seed} round{rnd}/{ROUNDS} best={best_cost} valid={valid}{meta_note} "
              f"usage={usage}", flush=True)

    return {
        "seed": seed,
        "model": MODEL,
        "rounds_done": ROUNDS,
        "best_initial": init_best,
        "best_final": pool[0][1],
        "best_route": f"<trace>{','.join(map(str, pool[0][0]))}</trace>",
        "round_costs": [r["best_cost"] for r in rows],
        "valid_rounds": sum(r["valid"] for r in rows),
        "meta_prompt_final": meta_prompt,
        "usage_total": usage_total,
        "solution_calls": solution_calls,
        "meta_calls": meta_calls,
        "rows": rows,
    }


def write_instance_files(seed, r):
    base = os.path.join(OUT_DIR, f"inst_seed{seed}")
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(r["rows"][0].keys()))
        w.writeheader()
        w.writerows(r["rows"])
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in r.items() if k != "rows"}, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- LMEA 结果读取与对比
def load_lmea():
    out = {}
    for f in os.listdir(LMEA_DIR):
        m = re.match(r"inst_seed(\d+)\.json$", f)
        if not m:
            continue
        seed = int(m.group(1))
        with open(os.path.join(LMEA_DIR, f), encoding="utf-8") as fh:
            d = json.load(fh)
        csv_path = os.path.join(LMEA_DIR, f"inst_seed{seed}.csv")
        valid_ratio = None
        if os.path.exists(csv_path):
            with open(csv_path, encoding="utf-8") as cf:
                rows = list(csv.DictReader(cf))
            nv = [int(r["n_valid"]) for r in rows]
            valid_ratio = sum(nv) / (len(nv) * 5.0)  # LMEA 每代 5 个候选解
        out[seed] = {"gen_costs": d["gen_costs"], "best_initial": d["best_initial"],
                     "best_final": d["best_final"], "valid_ratio": valid_ratio,
                     "model": d.get("model", "glm-4.5-air")}
    return out


def make_plots(opro, lmea):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 图 1：OPRO 收敛曲线（20 实例均值±std）
    curves = [r["round_costs"] for r in opro.values()]
    n_r = ROUNDS
    arr = [[c[i] for c in curves] for i in range(n_r)]
    mean = [sum(x) / len(x) for x in arr]
    std = [ (sum((v - m) ** 2 for v in x) / len(x)) ** 0.5 for x, m in zip(arr, mean)]
    xs = list(range(1, n_r + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(xs, mean, "b-", label="OPRO 均值")
    plt.fill_between(xs, [m - s for m, s in zip(mean, std)],
                     [m + s for m, s in zip(mean, std)], alpha=0.2, color="b", label="±std")
    plt.xlabel("轮次"); plt.ylabel("最佳路径长度")
    plt.title(f"OPRO 收敛曲线（TSP20, {len(curves)} 实例均值±标准差, glm-4.5-air）")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "opro_convergence.png"), dpi=150); plt.close()

    # 图 2：OPRO vs LMEA（共同 seed 1..8）
    common = sorted(set(opro) & set(lmea))
    if common:
        oc = [opro[s]["round_costs"] for s in common]
        lc = [lmea[s]["gen_costs"] for s in common]
        def ms(curves):
            L = max(len(c) for c in curves)
            arr = [[c[i] for c in curves if i < len(c)] for i in range(L)]
            m = [sum(x) / len(x) for x in arr]
            s = [(sum((v - mm) ** 2 for v in x) / len(x)) ** 0.5 for x, mm in zip(arr, m)]
            return m, s
        om, os_ = ms(oc)
        lm, ls = ms(lc)
        plt.figure(figsize=(9, 5.5))
        plt.plot(range(1, len(lm) + 1), lm, "r-", label=f"LMEA 均值（LLM 当进化算子, {len(common)} 实例, 100 代）")
        plt.fill_between(range(1, len(lm) + 1), [a - b for a, b in zip(lm, ls)],
                         [a + b for a, b in zip(lm, ls)], alpha=0.15, color="r")
        plt.plot(range(1, len(om) + 1), om, "b-", label=f"OPRO 均值（LLM 直接迭代改进解, {len(common)} 实例, 50 轮）")
        plt.fill_between(range(1, len(om) + 1), [a - b for a, b in zip(om, os_)],
                         [a + b for a, b in zip(om, os_)], alpha=0.2, color="b")
        plt.xlabel("代数 / 轮次"); plt.ylabel("最佳路径长度")
        plt.title("两种 LLM 优化范式对比：OPRO vs LMEA（TSP20, glm-4.5-air）")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "opro_vs_lmea.png"), dpi=150); plt.close()

        # 图 3：有效解率对比
        o_rate = sum(r["valid_rounds"] for r in opro.values() if r["seed"] in common) / (len(common) * ROUNDS)
        l_rate = sum(lmea[s]["valid_ratio"] or 0 for s in common) / len(common)
        plt.figure(figsize=(6, 4.5))
        plt.bar(["LMEA\n(每代5候选)", "OPRO\n(每轮1解)"], [l_rate, o_rate], color=["r", "b"], alpha=0.7)
        plt.ylabel("有效解率")
        plt.title("有效解率对比（共同实例均值, glm-4.5-air）")
        for i, v in enumerate([l_rate, o_rate]):
            plt.text(i, v + 0.01, f"{v:.1%}", ha="center")
        plt.ylim(0, 1.1)
        plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "valid_rate_compare.png"), dpi=150); plt.close()
    return common


def write_summary(opro, lmea, common):
    path = os.path.join(OUT_DIR, "summary_opro_vs_lmea.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "lmea_init", "lmea_final", "lmea_improve_pct",
                    "opro_init", "opro_final", "opro_improve_pct",
                    "opro_final_minus_lmea_final", "opro_valid_rate", "lmea_valid_rate"])
        for s in sorted(opro):
            r = opro[s]
            li = lmea.get(s)
            row = [s, "", "", "", r["best_initial"], r["best_final"],
                   round(100 * (r["best_initial"] - r["best_final"]) / r["best_initial"], 2),
                   "", round(r["valid_rounds"] / ROUNDS, 4), ""]
            if li:
                row[1] = li["best_initial"]; row[2] = li["best_final"]
                row[3] = round(100 * (li["best_initial"] - li["best_final"]) / li["best_initial"], 2)
                row[7] = round(r["best_final"] - li["best_final"], 2)
                row[9] = round(li["valid_ratio"], 4) if li["valid_ratio"] is not None else ""
            w.writerow(row)
    return path


def write_readme(opro, lmea, common, tokens):
    li = {s: lmea[s] for s in common}
    o_final = [opro[s]["best_final"] for s in sorted(opro)]
    o_init = [opro[s]["best_initial"] for s in sorted(opro)]
    o_imp = [100 * (a - b) / a for a, b in zip(o_init, o_final)]
    lines = []
    lines.append("# OPRO 复现：LLM 直接迭代改进解范式（vs LMEA 进化算子范式）\n")
    lines.append(f"- 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- 模型：glm-4.5-air（thinking disabled, max_tokens=512），3 个 key 轮询")
    lines.append(f"- 实例：TSP20，seed=1..20，坐标生成与 LMEA 完整版完全一致（`random.Random(seed)`, `round(uniform(0,100),2)`）")
    lines.append(f"- 每实例 {ROUNDS} 轮；解池初始 {INIT_POOL} 个随机解，保留 top-{POOL_K}；每 {META_EVERY} 轮做一次提示词自优化调用\n")
    lines.append("## 方法（OPRO, Optimization by Prompting, Yang et al. ICLR 2024, arXiv:2309.03409）\n")
    lines.append("- **范式定位**：OPRO 中 LLM 不作为进化算子，而是直接读取\"优化问题 + 当前候选解及其分数 + 历史\"，每轮提出一个更好的解；"
                 "同时周期性让 LLM 根据历史表现改写优化提示词（meta-prompt optimization），这是 OPRO 区别于普通\"LLM 求解 TSP\"的关键。")
    lines.append("- **与 LMEA 的辨别**：LMEA（LLM 当进化算子）中 LLM 输出被当作交叉/变异结果填入种群，由 GA 框架选择；"
                 "OPRO 中没有种群与遗传算子，只有\"解池 top-k + 提示词演化\"，选择压力来自\"保留 top-k 解池\"而非适应度排序配对。\n")
    lines.append("## 命令\n")
    lines.append("```bash")
    lines.append('export ZHIPU_KEYS="<key1>,<key2>,<key3>"   # key 仅走环境变量，严禁落盘')
    lines.append("E:/software/miniforge/python.exe opro_full.py > results/run.log 2>&1 &")
    lines.append("```\n")
    lines.append("## 结果\n")
    lines.append(f"| 指标 | OPRO（{len(opro)} 实例） | LMEA 完整版（{len(lmea)} 实例） |")
    lines.append("|---|---|---|")
    if common:
        li_final = [li[s]["best_final"] for s in common]
        li_imp = [100 * (li[s]["best_initial"] - li[s]["best_final"]) / li[s]["best_initial"] for s in common]
        o_rate = sum(opro[s]["valid_rounds"] for s in common) / (len(common) * ROUNDS)
        l_rate = sum(li[s]["valid_ratio"] or 0 for s in common) / len(common)
        lines.append(f"| 初始 best（均值） | {sum(o_init)/len(o_init):.2f} | {sum(li[s]['best_initial'] for s in common)/len(common):.2f} |")
        lines.append(f"| 最终 best（均值±std） | {sum(o_final)/len(o_final):.2f}±{(sum((x-sum(o_final)/len(o_final))**2 for x in o_final)/len(o_final))**0.5:.2f} | {sum(li_final)/len(li_final):.2f}±{(sum((x-sum(li_final)/len(li_final))**2 for x in li_final)/len(li_final))**0.5:.2f} |")
        lines.append(f"| 改进率（均值） | {sum(o_imp)/len(o_imp):.1f}% | {sum(li_imp)/len(li_imp):.1f}% |")
        lines.append(f"| 有效解率（共同实例） | {o_rate:.1%} | {l_rate:.1%} |")
    lines.append("")
    lines.append("逐实例结果见 `summary_opro_vs_lmea.csv`；曲线图：`opro_convergence.png`（OPRO 20 实例）、"
                 f"`opro_vs_lmea.png`（两种范式，共同 seed {common[0]}-{common[-1]}）、`valid_rate_compare.png`（有效率）。\n")
    lines.append("## Tokens 与费用\n")
    p, c, cd, t = tokens["prompt_tokens"], tokens["completion_tokens"], tokens["cached_tokens"], tokens["total_tokens"]
    lines.append(f"- 累计 tokens：input(prompt)={p:,}（其中 cached={cd:,}），output(completion)={c:,}，total={t:,}")
    lines.append(f"- 费用估算（glm-4.5-air：输入 0.8 元/M、输出 2 元/M）："
                 f"输入 {p/1e6*0.8:.2f} 元 + 输出 {c/1e6*2:.2f} 元 ≈ **{p/1e6*0.8 + c/1e6*2:.2f} 元**（未计缓存折扣）\n")
    lines.append("## 结论：两种 LLM 优化范式的辨别\n")
    if common:
        o_avg = sum(opro[s]["best_final"] for s in common) / len(common)
        l_avg = sum(li[s]["best_final"] for s in common) / len(common)
        winner = "OPRO" if o_avg < l_avg else "LMEA"
        lines.append(f"1. **范式机制不同**：LMEA 把 LLM 当作 GA 循环里的进化算子（种群 + 选择驱动），"
                     f"OPRO 让 LLM 直接迭代改进解并由提示词演化引导搜索（解池 + meta-prompt 驱动）。")
        lines.append(f"2. **最终质量**：共同 {len(common)} 实例上 OPRO 均值 {o_avg:.2f} vs LMEA 均值 {l_avg:.2f}，本设置下 {winner} 更优"
                     f"（注意轮数不同：LMEA 100 代 × 每代 5 候选 vs OPRO 50 轮 × 每轮 1 解，LLM 调用次数 {len(common)*100} vs {len(common)*ROUNDS+len(common)*4}）。")
        lines.append("3. **有效解率**：OPRO 每轮仅 1 个解，格式遵循失败即整轮无效；LMEA 每代 5 个候选可部分容错。")
        lines.append("4. **改进空间**：LMEA 依赖初始种群多样性，后期常停滞；OPRO 的提示词自优化提供了另一条搜索路径，"
                     "但单解提案的探索能力弱于 LMEA 的多解算子输出。")
    lines.append("\n## 遗留事项")
    lines.append(f"- LMEA 完整版当前仅含 seed {min(lmea)}-{max(lmea)} 的实例结果（目录 `lmea-repro/results/full/`），跨范式对比基于共同 {len(common)} 实例；"
                 f"若 LMEA 补齐 seed {max(lmea)+1}-20，可重跑 `make_plots`/`write_summary` 部分刷新对比。")
    lines.append("- OPRO 提示词自优化每 10 轮仅 1 次（共 4 次/实例），meta-prompt 质量对结果的影响未做消融。")
    lines.append("- 未做 2-opt 等局部搜索混合（保持纯范式对比）。")
    with open(os.path.join(OUT_DIR, "README_OPRO.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- 主流程
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    keys = load_keys()
    print(f"[启动] 模型={MODEL}（白名单校验通过），key 数={len(keys)}，实例=20，轮数={ROUNDS}", flush=True)
    client = LLMClient(keys, MODEL)
    opro = {}
    for seed in range(1, 21):
        r = run_instance(client, seed)
        write_instance_files(seed, r)
        opro[seed] = r
        print(f"[{MODEL}] 实例 seed={seed} 完成: init={r['best_initial']} -> final={r['best_final']} "
              f"valid={r['valid_rounds']}/{ROUNDS} usage={r['usage_total']}", flush=True)

    lmea = load_lmea()
    common = make_plots(opro, lmea)
    write_summary(opro, lmea, common)
    tokens = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "total_tokens": 0}
    for r in opro.values():
        for k in tokens:
            tokens[k] += r["usage_total"][k]
    write_readme(opro, lmea, common, tokens)
    with open(os.path.join(OUT_DIR, "DONE"), "w", encoding="utf-8") as f:
        f.write(f"done {datetime.now().isoformat()}\n")
    print("[完成] 全部 20 实例完成，DONE 已写入", flush=True)


if __name__ == "__main__":
    main()
