# -*- coding: utf-8 -*-
"""LLM 超参建议器（智谱 GLM-4.5-air，OpenAI 兼容，多 key 轮询）。

护栏：
- key 只从环境变量 ZHIPU_API_KEY / ZHIPU_API_KEY2 读取（可多个，round-robin），绝不落盘；
- 模型白名单硬校验 {"glm-4.5-air"}，不在白名单直接报错；
- max_tokens=512；thinking 通过 extra_body={"thinking":{"type":"disabled"}} 关闭；
- 调用节流：全局锁保证每次调用发起间隔 1~1.5s；429 后指数退避（1→2→4→8s 封顶，最多 6 次）；
- 输出要求 JSON（8 组超参 + ≤12 字 reason）；解析失败带反馈重试 2 次，仍失败随机回退（记录 fallback）。
"""
import json
import os
import random
import re
import threading
import time

import openai

from evaluator import BATCH, SPACE, random_suggestion

MODEL = "glm-4.5-air"
ALLOWED_MODELS = {"glm-4.5-air"}
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MAX_TOKENS = 512
BACKOFF_CAP = 8.0          # 429 退避上限
MAX_429_RETRIES = 6        # 429 最多退避重试次数
THROTTLE_MIN, THROTTLE_MAX = 1.0, 1.5

_lock = threading.Lock()
_clients = []
_key_idx = 0
_last_call = 0.0


def _load_keys():
    """从环境变量读取 key 轮询池：ZHIPU_API_KEY, ZHIPU_API_KEY2 ~ ZHIPU_API_KEY5。"""
    names = ["ZHIPU_API_KEY"] + [f"ZHIPU_API_KEY{i}" for i in range(2, 6)]
    keys = [os.environ[n] for n in names if os.environ.get(n)]
    if not keys:
        raise RuntimeError("未设置 ZHIPU_API_KEY* 环境变量（必须通过环境变量提供，禁止写进文件）")
    return keys


def _get_clients():
    global _clients
    if not _clients:
        for k in _load_keys():
            _clients.append(openai.OpenAI(api_key=k, base_url=BASE_URL,
                                          max_retries=0, timeout=120))
    return _clients


def _next_client():
    global _key_idx
    clients = _get_clients()
    with _lock:
        c = clients[_key_idx % len(clients)]
        _key_idx += 1
    return c


def _throttle():
    """全局节流：两次调用发起间隔 1~1.5s（持锁 sleep，多线程排队发起）。"""
    global _last_call
    with _lock:
        now = time.time()
        wait = _last_call + random.uniform(THROTTLE_MIN, THROTTLE_MAX) - now
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()


def _chat_with_backoff(messages):
    """一次对话请求：round-robin 换 key + 节流 + 429 指数退避（最多 6 次）。"""
    last_err = None
    for attempt in range(MAX_429_RETRIES + 1):
        try:
            client = _next_client()
            _throttle()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                extra_body={"thinking": {"type": "disabled"}},
                response_format={"type": "json_object"},
            )
            return resp
        except openai.RateLimitError as e:
            last_err = e
            wait = min(BACKOFF_CAP, 2.0 ** attempt)
            print(f"[429] 第{attempt + 1}/{MAX_429_RETRIES}次退避 {wait:.0f}s", flush=True)
            time.sleep(wait)
        except Exception as e:
            last_err = e
            wait = min(BACKOFF_CAP, 2.0 ** attempt)
            print(f"[retry] 调用异常 {e!r} 退避 {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"API 调用退避 {MAX_429_RETRIES} 次后仍失败: {last_err!r}")


SYSTEM_PROMPT = f"""你是超参数优化专家。我们正在用遗传算法(GA)求解旅行商问题(TSP, 20 个城市)，需要你为 GA 推荐超参数，目标是让 GA 得到的最优路径成本（tour length，越小越好）尽可能低。

GA 超参数搜索空间：
- population(种群规模): 整数 20 ~ 200
- crossover_rate(交叉率): 浮点数 0.5 ~ 1.0
- mutation_rate(变异率): 浮点数 0.05 ~ 0.5
- generations(进化代数): 整数 100 ~ 500

用户会给你一张表：已尝试的超参数组合 + 对应的 GA 最优路径成本（越低越好）。
请基于这些历史结果做 in-context learning，推断哪些超参数区域表现更好，然后推荐下一批 {BATCH} 组新的超参数组合。

要求：
1. 共 {BATCH} 组，尽量彼此不同、有代表性；既要有倾向性地靠近历史表现好的区域，也要保留一定探索性；
2. 每组超参数必须严格落在上述范围内（population、generations 为整数）；
3. 每组附一句 reason，不超过 12 个汉字，说明选择依据；
4. 只输出一个 JSON 对象，禁止输出任何其他文字、注释或 markdown 代码块，保持紧凑（总输出须在 512 token 内）；
   格式严格如下：
{{"suggestions": [{{"population": 120, "crossover_rate": 0.8, "mutation_rate": 0.15, "generations": 300, "reason": "依据第X组最优"}}]}}"""


def build_user_prompt(history, instance_idx, round_idx):
    if history:
        rows = "\n".join(
            f"| {i + 1} | {r['population']} | {r['crossover_rate']:.3f} | {r['mutation_rate']:.3f} "
            f"| {r['generations']} | {r['best_cost']:.4f} |"
            for i, r in enumerate(history)
        )
        tbl = "以下是已尝试的超参数及对应 GA 最优路径成本（tour length，越低越好）：\n\n| 序号 | population | crossover_rate | mutation_rate | generations | 最优成本 |\n|---|---|---|---|---|---|\n" + rows
    else:
        tbl = "目前还没有任何历史数据（这是第 1 轮）。请基于搜索空间做多样化初始化探索，覆盖不同区域。"

    return (
        f"当前任务：第 {instance_idx + 1} 个 TSP20 实例，第 {round_idx + 1} 轮优化（每轮建议 {BATCH} 组）。\n\n"
        f"{tbl}\n\n"
        f"请推荐下一批 {BATCH} 组超参数，输出为 JSON：{{\"suggestions\": [...]}}。"
    )


def extract_json(text):
    """尽力从模型输出中抽取 JSON 对象。"""
    if not text:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(t[a:b + 1])
        except Exception:
            pass
    return None


def validate(h):
    """校验并规范化一组超参；不合法返回 None。"""
    try:
        out = {
            "population": int(round(float(h.get("population")))),
            "crossover_rate": float(h.get("crossover_rate")),
            "mutation_rate": float(h.get("mutation_rate")),
            "generations": int(round(float(h.get("generations")))),
            "reason": str(h.get("reason", "")).strip(),
        }
    except (TypeError, ValueError):
        return None
    if not (SPACE["population"][0] <= out["population"] <= SPACE["population"][1]):
        return None
    if not (SPACE["crossover_rate"][0] <= out["crossover_rate"] <= SPACE["crossover_rate"][1]):
        return None
    if not (SPACE["mutation_rate"][0] <= out["mutation_rate"] <= SPACE["mutation_rate"][1]):
        return None
    if not (SPACE["generations"][0] <= out["generations"] <= SPACE["generations"][1]):
        return None
    out["crossover_rate"] = round(out["crossover_rate"], 3)
    out["mutation_rate"] = round(out["mutation_rate"], 3)
    return out


def _fallback_suggestions(instance_idx, round_idx):
    rng = random.Random(instance_idx * 100000 + round_idx)
    return [random_suggestion(rng) for _ in range(BATCH)]


def suggest(history, instance_idx, round_idx):
    """调用 LLM 建议下一批 BATCH 组超参。

    返回: (suggestions, raw, usage_dict, fallback)
      suggestions: list[dict]（含 population/crossover_rate/mutation_rate/generations/reason）
      raw: 模型原始回复；usage_dict: {prompt_tokens, completion_tokens}
      fallback: bool，是否因解析失败而用了随机回退
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(history, instance_idx, round_idx)},
    ]
    last_content = ""
    last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for _attempt in range(3):  # 初次 + 最多 2 次带反馈的 JSON 重试
        resp = _chat_with_backoff(messages)
        content = resp.choices[0].message.content or ""
        last_content = content
        if resp.usage is not None:
            last_usage = {
                "prompt_tokens": int(resp.usage.prompt_tokens or 0),
                "completion_tokens": int(resp.usage.completion_tokens or 0),
            }

        obj = extract_json(content)
        raw_sugs = obj.get("suggestions") if isinstance(obj, dict) else None
        if isinstance(raw_sugs, list) and len(raw_sugs) >= BATCH:
            valid = []
            for h in raw_sugs:
                v = validate(h)
                if v is not None:
                    valid.append(v)
            if len(valid) >= BATCH:
                return valid[:BATCH], content, last_usage, False

        # 解析/校验失败 → 把错误反馈给模型再试一次
        err = (f"无法从你的输出解析出 {BATCH} 组合法超参（reason 请控制在 12 个汉字内）。"
               f"请重新只输出严格 JSON：{{\"suggestions\":[{{\"population\":100,\"crossover_rate\":0.8,"
               f"\"mutation_rate\":0.15,\"generations\":300,\"reason\":\"依据第X组最优\"}}]}}，共 {BATCH} 组。")
        messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": err},
        ]

    # 彻底失败 → 随机回退，保证实验不中断
    return _fallback_suggestions(instance_idx, round_idx), last_content, last_usage, True
