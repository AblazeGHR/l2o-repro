# OPRO 复现：LLM 直接迭代改进解范式（vs LMEA 进化算子范式）

- 运行时间：2026-08-28 13:09
- 模型：glm-4.5-air（thinking disabled, max_tokens=512），3 个 key 轮询
- 实例：TSP20，seed=1..20，坐标生成与 LMEA 完整版完全一致（`random.Random(seed)`, `round(uniform(0,100),2)`）
- 每实例 50 轮；解池初始 3 个随机解，保留 top-5；每 10 轮做一次提示词自优化调用

## 方法（OPRO, Optimization by Prompting, Yang et al. ICLR 2024, arXiv:2309.03409）

- **范式定位**：OPRO 中 LLM 不作为进化算子，而是直接读取"优化问题 + 当前候选解及其分数 + 历史"，每轮提出一个更好的解；同时周期性让 LLM 根据历史表现改写优化提示词（meta-prompt optimization），这是 OPRO 区别于普通"LLM 求解 TSP"的关键。
- **与 LMEA 的辨别**：LMEA（LLM 当进化算子）中 LLM 输出被当作交叉/变异结果填入种群，由 GA 框架选择；OPRO 中没有种群与遗传算子，只有"解池 top-k + 提示词演化"，选择压力来自"保留 top-k 解池"而非适应度排序配对。

## 命令

```bash
export ZHIPU_KEYS="<key1>,<key2>,<key3>"   # key 仅走环境变量，严禁落盘
E:/software/miniforge/python.exe opro_full.py > results/run.log 2>&1 &
```

## 结果

| 指标 | OPRO（20 实例） | LMEA 完整版（13 实例） |
|---|---|---|
| 初始 best（均值） | 987.62 | 842.76 |
| 最终 best（均值±std） | 477.90±50.62 | 430.34±62.29 |
| 改进率（均值） | 51.1% | 48.6% |
| 有效解率（共同实例） | 99.7% | 56.4% |

逐实例结果见 `summary_opro_vs_lmea.csv`；曲线图：`opro_convergence.png`（OPRO 20 实例）、`opro_vs_lmea.png`（两种范式，共同 seed 1-13）、`valid_rate_compare.png`（有效率）。

## Tokens 与费用

- 累计 tokens：input(prompt)=952,084（其中 cached=623,968），output(completion)=58,556，total=1,010,640
- 费用估算（glm-4.5-air：输入 0.8 元/M、输出 2 元/M）：输入 0.76 元 + 输出 0.12 元 ≈ **0.88 元**（未计缓存折扣）

## 结论：两种 LLM 优化范式的辨别

1. **范式机制不同**：LMEA 把 LLM 当作 GA 循环里的进化算子（种群 + 选择驱动），OPRO 让 LLM 直接迭代改进解并由提示词演化引导搜索（解池 + meta-prompt 驱动）。
2. **最终质量**：共同 13 实例上 OPRO 均值 482.96 vs LMEA 均值 430.34，本设置下 LMEA 更优（注意轮数不同：LMEA 100 代 × 每代 5 候选 vs OPRO 50 轮 × 每轮 1 解，LLM 调用次数 1300 vs 702）。
3. **有效解率**：OPRO 每轮仅 1 个解，格式遵循失败即整轮无效；LMEA 每代 5 个候选可部分容错。
4. **改进空间**：LMEA 依赖初始种群多样性，后期常停滞；OPRO 的提示词自优化提供了另一条搜索路径，但单解提案的探索能力弱于 LMEA 的多解算子输出。

## 遗留事项
- LMEA 完整版当前仅含 seed 1-13 的实例结果（目录 `lmea-repro/results/full/`），跨范式对比基于共同 13 实例；若 LMEA 补齐 seed 14-20，可重跑 `make_plots`/`write_summary` 部分刷新对比。
- OPRO 提示词自优化每 10 轮仅 1 次（共 4 次/实例），meta-prompt 质量对结果的影响未做消融。
- 未做 2-opt 等局部搜索混合（保持纯范式对比）。
