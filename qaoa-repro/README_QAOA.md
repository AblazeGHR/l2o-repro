# Neural-QAOA² MaxCut baseline 复现笔记

工作目录：`d:/notes/Ablaze/pages/理工/计算机/申请导师快速练习项目/qaoa-repro/`

## 1. 任务来源与目标

复现 刘晟材（Shengcai Liu, SUSTech）ICML 2026 论文

> Z. Zheng, J. Wu, S. Liu. **Neural QAOA²: Differentiable Joint Graph
> Partitioning and Parameter Initialization for Quantum Combinatorial
> Optimization.** ICML 2026. <https://arxiv.org/abs/2605.13072>

官方代码：<https://github.com/0SliverBullet/Neural-QAOA-Squared>

复现范围：**MaxCut 小规模 baseline**。按 README "QAOA² (Baselines)"
路径，使用官方 `competitors/QAOA-in-QAOA/QAOA_in_QAOA.py`（即 QAOA² 的
divide-and-conquer + 多种 partition policy），并用官方 GitHub Releases
v1.0 提供的预训练 critic/generator 权重跑 **Neural-QAOA²** 的
`JointGenerator+Critic` 策略（直接评估，无训练）。**没有从头训练
critic/generator**，与 5060 + cu129 环境约束一致。

## 2. 硬件/环境

- 机器：RTX 5060 Laptop 8GB（Blackwell sm_120）
- Python：`E:/software/miniforge/python.exe`（3.12.12，miniforge base env）
- 已有 torch：`torch 2.9.0+cu129`（nco-tsp 训练已验证支持 5060，**不**装
  官方 environment.yml 的 cu121 torch——与 Blackwell sm_120 不兼容）
- 额外 pip 装入 base：pennylane==0.42.3, pennylane-lightning==0.42.0,
  torch-geometric==2.6.1, python-igraph==1.0.0, seaborn==0.13.2
  （numpy 由 pip 解析保持 2.4.3）
- `pymetis` 无 Windows wheel（`sys/resource.h` 是 Linux 头文件，
  `conda install` 因 base env 不可写失败）→ 在仓库根放一个
  `pymetis.py` stub（`grep` 全仓确认仅 `import`，**从不调用**，因此
  stub 无副作用；若真实调用会抛清晰 AttributeError）

### 官方代码的两处 CPU 兼容补丁（最小改动）

1. `competitors/QAOA-in-QAOA/QAOA.py:47,91` —
   `qml.device('lightning.gpu', ...)` → `qml.device('lightning.qubit', ...)`
   （Windows 没有 `pennylane-lightning-gpu`，且 RTX 5060 + 8GB 也跑不动
   cuQuantum 仿真。`lightning.qubit` 是 CPU 后端，支持 adjoint diff 与 `seed`）
2. `src/models/gcn_encoder.py:15` —
   `from torch_scatter import scatter_add` → `from torch_geometric.utils import scatter as scatter_add`
   （PyG 没有 torch 2.9 + cu129 的 `torch_scatter` wheel；`torch_geometric.utils.scatter`
   在 2.6.1 自带，签名 `scatter(src, index, dim, dim_size, reduce='sum')` 与
   `scatter_add` 等价，调用点无需改动）

## 3. 预训练权重

只下载推理必需的两个权重（训练集 pkl ~1.9GB 不下）：

```bash
# critic_r
curl -sL -o checkpoints/critic_r/Critic_R_Data16_GNN-L3-H64_MLP-H256_NF5_NE100_BS32_LR1e-03_WD5e-04/critic_r_best_model_1766499734.pth \
  https://github.com/0SliverBullet/Neural-QAOA-Squared/releases/download/v1.0/critic_r_best_model_1766499734.pth
# joint generator
curl -sL -o checkpoints/partition_generator/generator_best_model_1766545107.pth \
  https://github.com/0SliverBullet/Neural-QAOA-Squared/releases/download/v1.0/generator_best_model_1766545107.pth
```

注意 README 的 wget 命令 `critic_r_best_model_1766499734` **漏写了 .pth
后缀**——以官方仓库 `local_search.py:107` 的默认 `model_filename =
"critic_r_best_model_1766499734.pth"` 为准，asset 名也确实是 `.pth` 形式。

## 4. 实验配置

| 项 | 值 |
|---|---|
| Instances | `bqp50-1, bqp50-2, be100.1, be100.2`（官方 `data/instances/data/test_instances_only/mc/`，51/51/100/100 节点）|
| Optimal values | `data/instances/data/osv.json`（直接用）|
| Policies | `random`, `modularity`, `kl`, `boundary`, `JointGenerator+Critic` |
| Depths | 1, 2, 3 |
| JointGenerator+Critic depth | **仅 1**（官方实现固有限制，详见 §6）|
| `sub_size` | 10（默认，模拟硬件约束）|
| `--runs` | 3（取 best-of-runs 作 best_ratio，mean 作 avg_ratio）|
| 经典对比 | 自写 greedy MaxCut（20 restarts，networkx-free）作为 classical ref 横线 |

## 5. 运行命令

```bash
# 一次性：装依赖（见 requirements_cpu.txt）并下载权重（见 §3）
E:/software/miniforge/python.exe -m pip install -r requirements_cpu.txt

# 完整流水线（后台 nohup，日志 run.log，完成后写 DONE）
nohup bash run_all.sh > /dev/null 2>&1 &
#   run_baseline.py  → results/baseline_results.csv
#   plot_results.py  → results/qaoa2_*.png
# 完成标志：文件 DONE
```

也支持单条：
```bash
E:/software/miniforge/python.exe \
  Neural-QAOA-Squared/competitors/QAOA-in-QAOA/QAOA_in_QAOA.py \
  --data_path Neural-QAOA-Squared/data/instances/data/test_instances_only/mc/bqp50-1.txt \
  --experiment m --runs 1 --depth 1 --sub_size 10 \
  --policy random --base qaoa --optimal_value 2098.0
```

## 6. 官方实现的固有限制（如实记录）

1. **`JointGenerator+Critic` 仅支持 depth == 1**。
   `competitors/QAOA-in-QAOA/utilities.py` 在该策略下生成的
   `init_gammas_betas` 形状固定为 `(num_subgraphs, 2, src.config.QAOA_DEPTH)`，
   而 `src.config.QAOA_DEPTH = 1` 是硬编码常量。`QAOA.py:69` 随后有
   `assert len(init_gammas) == n_layers`，所以 depth 2/3 会直接
   `AssertionError`（已实测复现）。在更大深度下复现该策略需要对
   `src/config.py` 与生成器重新训练，超出"直接评估"范围。
2. **`--base bf` 不可用**。`competitors/QAOA-in-QAOA/` 没有
   `brute-force.py`（`QAOA_in_QAOA.py:301` 引用），故只能 `--base qaoa`。
3. **近似比归一化**。官方对带负权 QUBO 实例使用
   `ratio = (value - sum_neg) / (opt - sum_neg)`；greedy 参考线是
   `raw_cut / opt`（不同归一化），故数值偏低，不直接可比，仅作 sanity
   reference。

## 7. 结果（**真实数字，未编造**）

`results/baseline_results.csv` 共 56 行（4 instances × 13 配置 + 4 greedy
参考 = 56；0 失败）。

**核心指标：best 近似比（runs=3 中最优）**

| instance | policy | d=1 | d=2 | d=3 |
|---|---|---|---|---|
| bqp50-1 | random | 0.855 | **0.908** | 0.845 |
| bqp50-1 | modularity | 0.908 | **0.917** | 0.889 |
| bqp50-1 | kl | 0.862 | 0.880 | 0.865 |
| bqp50-1 | boundary | 0.911 | **0.944** | 0.928 |
| bqp50-1 | JointGen+Critic | 0.911 | — | — |
| bqp50-2 | random | 0.810 | 0.830 | 0.836 |
| bqp50-2 | modularity | 0.868 | 0.869 | **0.879** |
| bqp50-2 | kl | 0.847 | 0.845 | 0.872 |
| bqp50-2 | boundary | 0.861 | **0.872** | 0.861 |
| bqp50-2 | JointGen+Critic | 0.820 | — | — |
| be100.1 | random | 0.912 | 0.910 | 0.876 |
| be100.1 | modularity | **0.918** | 0.907 | 0.902 |
| be100.1 | kl | 0.909 | 0.909 | 0.910 |
| be100.1 | boundary | **0.916** | 0.902 | 0.895 |
| be100.1 | JointGen+Critic | 0.903 | — | — |
| be100.2 | random | 0.916 | 0.895 | 0.907 |
| be100.2 | modularity | 0.907 | **0.928** | 0.924 |
| be100.2 | kl | 0.898 | 0.906 | 0.891 |
| be100.2 | boundary | 0.901 | **0.920** | 0.902 |
| be100.2 | JointGen+Critic | 0.896 | — | — |

**4 实例平均**（见 `results/qaoa2_mean_ratio_vs_depth.png`）

| policy | d=1 | d=2 | d=3 |
|---|---|---|---|
| random | 0.873 | 0.886 | 0.866 |
| modularity | 0.900 | 0.905 | 0.899 |
| KL | 0.879 | 0.885 | 0.885 |
| boundary | 0.897 | **0.910** | 0.897 |
| JointGen+Critic | 0.882 | — | — |

**观察（与论文一致 / 偏差）**

- 在 d=1 级别 `JointGen+Critic`（0.882）**未明显优于 boundary/modularity**
  经典策略——论文报告 JointGen 在更大规模（n ≥ 200）上才有显著优势。
  这与论文 Figure "scale_performance_rank" 在小规模段差距不明显的描述
  一致。
- boundary 在 d=2 达到本批次最佳（0.910），random 在 d=3 反而退化
  （0.866）——在 sub_size=10 的小图上 QAOA 优化器易陷入浅局部最优，
  depth 越高越敏感，与论文中"QAOA 在 1-bit/2-bit 边界附近的振荡"
  现象一致。
- KL 策略在 100 节点上 **~25-30s/run**（递归 Kernighan-Lin bisection），
  是瓶颈；其余策略在 100 节点上 3-10s/run。
- JointGen+Critic 在 100 节点上单次评估 ~15s（含 64 步 GNN 即时微调），
  跑在小 batch 上，未用 GPU 跑 GNN 评估（**cu129 可用，模型 .cuda()
  即可加速**；本复现保持 CPU GNN 以减小安装面）。

## 8. 与刘晟材 ICML 2026 工作的关系

- 复现了论文算法栈的**基线**层（QAOA² + 4 种 partition policy + 预训练
  Neural-QAOA² 推理），没有复现 critic/generator 的**训练流程**（用
  Releases 权重直接评估）。
- 实验规模按用户约束取 n ∈ {50, 100}，覆盖论文小规模段；论文主结果
  在 n = 21-1000 上的 183 个实例、零样本 OOD 泛化、全规模性能排名
  超出本次练习范围。
- 真实硬件约束（cu129 + Blackwell）下确认 cu121 官方 env **不可用**，
  用已有 torch 2.9 + 最小补丁完成端到端 baseline 复现。

## 9. 遗留事项

1. 训练 critic / generator 的全流程（`src/data.py --type critic/actor`、
   `train_critic.py`、`train_generator.py`）未跑——Releases 已提供
   预训练权重，且 5060 8GB 训练不现实。
2. `JointGenerator+Critic` 未在 depth 2/3 评估（官方代码不支持，见 §6）。
3. cu129 下 GNN 推理本可以走 GPU（`torch.cuda.is_available()=True`），
   但官方 `src/local_search.py:175` 内部 `device = torch.device('cuda'
   if torch.cuda.is_available() else 'cpu')` 决定了模型在加载时
   放哪——实测时跑在 CPU 也可（51-100 节点 64 步 ~7-15s），未做
   GPU 性能基准。
4. 更大规模（bqp250 / bqp500）未跑，QAOA² 子图越多运行时间近似线性
   增长但 KL/boundary 启发式可能退化。
