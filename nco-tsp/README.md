# 神经组合优化：Transformer + REINFORCE 求解 TSP

复现 Kool et al. (ICLR 2019) *"Attention, Learn to Solve Routing Problems!"*：
用 **Transformer 编码器 + attention pointer 解码器** 做策略网络，**REINFORCE 策略梯度 + greedy rollout baseline** 训练，
在随机 **TSP20** 实例上求解旅行商问题（坐标取自单位正方形均匀分布）。

## 方法（一句话）

Encoder 把节点坐标编码为 embedding，Decoder 自回归地用 masked attention 逐点选下一个城市，
用 rollout baseline（当前最优策略的贪婪解）作为 REINFORCE 的 baseline 降低方差，策略梯度最大化期望往返长度（取负为损失）。

## 代码来源

clone 官方仓库 `wouterkool/attention-learn-to-route`（master, commit c9abf41）。

> 做了两处必要的环境适配补丁（原仓库为 2019 年代码）：
> 1. `run.py`：`tensorboard_logger` 在 Python 3.12 / 新版 protobuf 下损坏，改为可选导入；
> 2. `utils/functions.py`：`torch.load` 加 `weights_only=False`（torch 2.6+ 默认值变更）。

## 环境

- Python: `E:/software/miniforge/python.exe` (3.12)
- PyTorch: `2.9.0+cu129`（CUDA，RTX 5060 Laptop 8GB）
- numpy / scipy / tqdm / matplotlib / ortools

## 训练命令

```bash
python run.py --problem tsp --graph_size 20 --baseline rollout \
  --batch_size 512 --epoch_size 204800 --n_epochs 80 \
  --val_size 2000 --eval_batch_size 1024 --log_step 50 \
  --checkpoint_epochs 10 --no_tensorboard --no_progress_bar \
  --run_name tsp20_rollout_gpu
```

- 每轮 400 batches（batch 512，共 204800 实例），80 轮共 ~16M 训练实例。
- 运行日志：`results/train_tsp20_gpu.log`

## 结果

- 收敛图：`results/training_curves.png`
- 对比表：`results/comparison_tsp20.csv`

| 方法 | 平均 tour length (TSP20) |
|---|---|
| Attention Model (greedy, 本文复现) | 3.8601 ± 0.3070 |
| Nearest Neighbour | 4.4680 ± 0.5351 |
| 2-opt (NN init) | 3.9307 ± 0.3503 |
| Random Insertion | 3.9984 ± 0.3577 |
| OR-Tools (GLS) | 3.8302 ± 0.2994 |
| 最优 / Kool et al. 2019 | 3.83 |

## 文件

- `run.py` / `train.py` — 官方训练入口（原样，仅依赖补丁）
- `plot_results.py` — 解析训练日志出收敛曲线图
- `eval_trained.py` — 在固定测试集上对比 AM / NN / 2-opt / RI / OR-Tools
- `README_official.md` — 官方原版 README
