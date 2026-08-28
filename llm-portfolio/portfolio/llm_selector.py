# -*- coding: utf-8 -*-
"""LLM 选择器：给定 TSP 实例特征，预测最优求解算法。

- 智谱 OpenAI 兼容接口，base_url=https://open.bigmodel.cn/api/paas/v4
- API key 只从环境变量 ZHIPU_API_KEY 读取（严禁写入代码/文件/日志）
- 模型硬白名单 {"glm-4.5-air"}，不在白名单直接报错
- thinking disabled（extra_body），max_tokens 固定 512
"""

import json
import os
import random
import re
import threading
import time

from openai import OpenAI

ALLOWED_MODELS = {"glm-4.5-air"}
ALGORITHM_POOL = ["GA", "GA2opt", "NN2opt", "RI2opt"]

BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MAX_TOKENS = 512
TEMPERATURE = 0.2

SYSTEM_PROMPT = """你是一名"算法组合（Algorithm Portfolio）"专家，任务是根据 TSP（旅行商问题）实例的数值特征，预测哪个求解算法在 2 秒时间预算（单 CPU）内能给出最短回路。

候选算法池（每个算法都有 2 秒预算，结果取预算内最佳）：
1. "GA"    —— 遗传算法（顺序交叉 OX + 交换变异，无局部搜索）。种群大、探索强，但对难收敛的小实例未必最快收敛。
2. "GA2opt"—— 遗传算法 + 对部分个体施加 2-opt 局部搜索。全局探索 + 局部精修，通常最稳健。
3. "NN2opt"—— 最近邻贪心建初始回路 + 2-opt 局部改进（每个城市作起点多起点）。对小实例（n<=30）常能极快收敛到最优。
4. "RI2opt"—— 随机插入建初始回路 + 2-opt（多次重启）。初始解质量中等，但重启次数多。

输入为实例特征 JSON，字段含义：
- n: 节点数
- mean_x/std_x/mean_y/std_y: 坐标均值与标准差
- bbox_width/height/diag: 包围盒宽高与对角线
- mean_nn_dist/min_nn_dist/max_nn_dist/std_nn_dist: 最近邻距离的均值/最小/最大/标准差
- mean_2nn_dist: 第二近邻距离均值
- nn_spread_ratio: 最近邻均值距离 / (1/sqrt(n))，>1 分散，<1 聚集
- nn_cv: 最近邻距离变异系数（几何均匀性）
- nn_tour_len / ri_tour_len: 最近邻/随机插入快速回路长度
- nn_over_ri: 两种快速回路长度之比
- quick_tour_per_node: 快速回路长度均值 / n（单位节点成本）

请结合几何结构、节点规模、特征分布给出判断。只输出一个 JSON 对象（不要输出任何其它文字、不要用 Markdown 代码块包裹）：
{"algorithm": "GA|GA2opt|NN2opt|RI2opt", "confidence": 0到1之间的小数, "reason": "不少于 80 字的详细分析：结合具体特征说明该实例的几何结构与难度，解释为什么所选算法最可能在 2 秒内胜出，并简述落选算法的劣势"}"""


def _extract_json(text):
    """从模型输出中鲁棒提取 JSON 对象。"""
    if not text:
        return None
    text = text.strip()
    # 去掉可能的 markdown 代码块围栏
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class LLMSelector:
    def __init__(self, model="glm-4.5-air", max_tokens=MAX_TOKENS,
                 temperature=TEMPERATURE, timeout=90, max_retries=3):
        # 硬白名单校验：不读环境变量里的模型名，不 fallback
        if model not in ALLOWED_MODELS:
            raise ValueError(f"模型不在白名单 {ALLOWED_MODELS} 中: {model!r}")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0,
                            "total_tokens": 0, "calls": 0}

        key = os.environ.get("ZHIPU_API_KEY")
        if not key:
            raise RuntimeError("未设置环境变量 ZHIPU_API_KEY")
        self.client = OpenAI(api_key=key, base_url=BASE_URL, timeout=timeout)

    def predict(self, features):
        """对单个实例预测最优算法。返回 dict 或 None（解析失败时）。"""
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",
                         "content": f"TSP 实例特征：\n{json.dumps(features, ensure_ascii=False)}"},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    extra_body={"thinking": {"type": "disabled"}},
                )
            except Exception as e:  # 网络/限流等可重试错误
                last_error = e
                time.sleep(2 ** attempt)
                continue

            usage = getattr(resp, "usage", None)
            if usage is not None:
                self.total_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                self.total_usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
                self.total_usage["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
                self.total_usage["calls"] += 1

            content = resp.choices[0].message.content if resp.choices else None
            obj = _extract_json(content)
            if obj is None:
                # 已计费但解析失败：不再重试（避免重复计费）
                return {"algorithm": None, "confidence": 0.0,
                        "reason": content, "parse_failed": True,
                        "usage": self._last_usage(resp)}
            alg = obj.get("algorithm")
            if alg not in ALGORITHM_POOL:
                return {"algorithm": None, "confidence": float(obj.get("confidence", 0) or 0),
                        "reason": content, "parse_failed": True,
                        "invalid_alg": alg, "usage": self._last_usage(resp)}
            return {
                "algorithm": alg,
                "confidence": float(obj.get("confidence", 0.5) or 0.5),
                "reason": str(obj.get("reason", "")),
                "parse_failed": False,
                "usage": self._last_usage(resp),
            }
        return {"algorithm": None, "confidence": 0.0,
                "reason": f"API error: {last_error}", "parse_failed": True}

    def _last_usage(self, resp):
        usage = getattr(resp, "usage", None)
        if usage is None:
            return None
        return {
            "prompt": getattr(usage, "prompt_tokens", 0) or 0,
            "completion": getattr(usage, "completion_tokens", 0) or 0,
            "total": getattr(usage, "total_tokens", 0) or 0,
        }


class _KeyRotator:
    """多 key 轮询器（线程安全，进程内单例）。

    key 只从环境变量读取（ZHIPU_API_KEYS 逗号分隔，退回 ZHIPU_API_KEY），
    绝不写入代码/文件/日志。
    """

    def __init__(self, keys):
        self.keys = keys
        self.clients = [OpenAI(api_key=k, base_url=BASE_URL, timeout=90) for k in keys]
        self._idx = 0
        self._lock = threading.Lock()
        self.stats = {"success": 0, "rate_limited": 0, "other_error": 0}

    def next_client(self):
        with self._lock:
            c = self.clients[self._idx % len(self.clients)]
            self._idx += 1
            return c

    def note(self, kind):
        with self._lock:
            self.stats[kind] += 1

    def stats_str(self):
        with self._lock:
            s = dict(self.stats)
        return (f"calls_ok={s['success']} calls_429={s['rate_limited']} "
                f"calls_other_err={s['other_error']}")


_rotator = None
_rotator_lock = threading.Lock()


def get_rotator():
    global _rotator
    with _rotator_lock:
        if _rotator is None:
            keys = [k.strip() for k in os.environ.get("ZHIPU_API_KEYS", "").split(",")
                    if k.strip()]
            if not keys:
                k = os.environ.get("ZHIPU_API_KEY")
                if k:
                    keys = [k]
            if not keys:
                raise RuntimeError("未设置环境变量 ZHIPU_API_KEYS / ZHIPU_API_KEY")
            _rotator = _KeyRotator(keys)
        return _rotator


def rotator_stats():
    return get_rotator().stats_str()


def _is_rate_limit_error(e):
    if getattr(e, "status_code", None) == 429:
        return True
    return "1302" in str(e) or "429" in str(e)[:80]


class ZhipuChat:
    """通用智谱 chat 客户端（供实例生成器/在线选择器复用）。

    - 多 key 轮询（每次调用换下一个 key），缓解账户级 429 限流
    - 每次调用前随机 sleep 0.3~0.8s；429 时退避 1~2.5s 后换 key 重试
    - 失败重试最多 max_retries 次；白名单硬校验；usage 逐调用累加
    """

    def __init__(self, model="glm-4.5-air", max_tokens=MAX_TOKENS,
                 temperature=TEMPERATURE, timeout=90, max_retries=5,
                 min_interval=0.3, max_interval=0.8):
        if model not in ALLOWED_MODELS:
            raise ValueError(f"模型不在白名单 {ALLOWED_MODELS} 中: {model!r}")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.usage = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
        self._last_call = 0.0
        self._rotator = get_rotator()

    def complete(self, messages, max_tokens=None):
        """发送消息，返回 (content, usage_dict)。重试耗尽则抛最后一次异常。"""
        max_tokens = max_tokens or self.max_tokens
        # 限流退避：两次调用之间至少间隔 0.3~0.8s
        now = time.monotonic()
        wait = random.uniform(self.min_interval, self.max_interval)
        if now - self._last_call < wait:
            time.sleep(wait - (now - self._last_call))
        self._last_call = time.monotonic()

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            client = self._rotator.next_client()  # 每次尝试轮换 key
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    extra_body={"thinking": {"type": "disabled"}},
                )
            except Exception as e:
                last_error = e
                if _is_rate_limit_error(e):
                    self._rotator.note("rate_limited")
                    time.sleep(random.uniform(1.0, 2.5))
                else:
                    self._rotator.note("other_error")
                    time.sleep((2 ** attempt) + random.uniform(0, 1))
                continue
            self._rotator.note("success")
            usage = getattr(resp, "usage", None)
            if usage is not None:
                self.usage["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
                self.usage["completion"] += getattr(usage, "completion_tokens", 0) or 0
                self.usage["total"] += getattr(usage, "total_tokens", 0) or 0
                self.usage["calls"] += 1
            content = resp.choices[0].message.content if resp.choices else None
            return content, self._last_usage(resp)
        raise RuntimeError(f"API 重试 {self.max_retries} 次仍失败: {last_error}")

    def _last_usage(self, resp):
        usage = getattr(resp, "usage", None)
        if usage is None:
            return None
        return {
            "prompt": getattr(usage, "prompt_tokens", 0) or 0,
            "completion": getattr(usage, "completion_tokens", 0) or 0,
            "total": getattr(usage, "total_tokens", 0) or 0,
        }
