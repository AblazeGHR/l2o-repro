# L2O-Repro：LLM / AI for Optimization 复现实验集

围绕「用机器学习 / 大语言模型改进优化」（**L2O, Learning to Optimize**）方向的系列复现与对照实验，全部基于真实运行结果，诚实标注进行中项。

## 子项目一览

| 子目录 | 方向 | 内容 | 状态 |
|---|---|---|---|
| `nco-tsp` | 神经组合优化 NCO | Kool et al. (ICLR'19) Attention Model 完整训练（80 epochs ≈16M 实例，验证 avg 3.8601 逼近论文 3.83）；评估阶段定位并修复 checkpoint 静默加载 bug | ✅ 完成 |
| `opro-repro` | LLM 优化器（范式对照） | OPRO（ICLR'24）vs LMEA 两种范式对比：LMEA 靠多候选大预算换更优解，OPRO 以一半调用量达 99.7% 有效解率 | ✅ 完成 |
| `llm-portfolio` | 算法组合 Algorithm Portfolios | LLM 生成定向难实例（分歧度 187%+）+ 在线算法选择（warm-up gap 0.039%）；含首轮"万能算法"失败归因与升级实验 | ✅ 完成 |
| `llm-hpo` | AutoML | LLM 作为超参数优化器 vs Optuna TPE / Random Search 同预算对比（30 实例 × 30 轮） | 🔄 分析进行中（诚实标注：LLM 未跑赢 TPE） |
| `qaoa-repro` | AI for Quantum | Neural-QAOA²（ICML'26）评估层复现：解决 Blackwell 架构兼容（cu121→cu129）与 Windows 缺 wheel 问题，结果与论文结论一致（优势在 n≥200 才显现） | ✅ 完成 |

## 相关独立仓库

- **lmea-repro** — LMEA（LLM as Evolutionary Optimizers, CEC'24）复现 + 温度机制改进实验
- **ga-tsp-visualizer** — GA + 2-opt 解 TSP 基线项目
- **pan** — 多 Agent 编排调度平台（以上实验的编排基础设施）

## 复现环境

- 训练：RTX 5060（Blackwell），torch 2.9 + cu129
- LLM：智谱 glm-4.5-air（API key 仅走环境变量，严禁落盘）
- 平台：Windows 11

## 诚实声明

- 每个子目录的结果数据均来自实际运行，失败的实验（nco 评估 bug、portfolio 首轮退化、hpo 未跑赢 TPE）全部保留并写入分析，不掩盖。
- 模型权重（`*.pt` 等）与运行日志不纳入版本控制，需复现时按各目录 README 运行命令生成。
