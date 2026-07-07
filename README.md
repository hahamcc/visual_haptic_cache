# Visual-Haptic Cache Project

本项目用于研究视觉到触觉预测中的缓存检索方法。

当前仓库处于重建阶段：之前的数据、代码、模型权重和实验输出已经丢失，但 `docs/` 中保留了论文阅读、阶段回顾和项目思路。当前目标不是立刻做复杂大模型，而是先恢复数据消失前的最小闭环。

## 项目目标

- 从视频帧中预测接触区域
- 提取运动特征，包括速度、方向、接触区域
- 建立视觉-触觉缓存库
- 在线检索缓存，减少触觉生成模型的计算开销

## 快速导航

- `AGENTS.md`: Codex 和人工协作规则，以及项目专属重建约束
- `Process.md`: 当前重建计划、阶段目标、验收标准和进度日志
- `docs/`: 历史论文阅读、思路整理、阶段回顾和前情资料
- `configs/default.yaml`: 默认配置入口，后续路径和阈值尽量放在这里

## 环境

本项目使用 Conda 环境 `haptic-cache`：

```bash
conda activate haptic-cache
```

环境配置默认使用 CUDA 版 PyTorch：

```bash
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
```

如果 `torch.version.cuda` 有版本号但 `torch.cuda.is_available()` 是 `False`，优先检查系统 NVIDIA driver：

```bash
nvidia-smi
```

如果需要重建环境，可以使用：

```bash
conda env create -f environment.yml
```

## 当前重建路线

第一阶段先恢复数据与标签基础：

- 建立 RGB/touch 帧索引
- 通过触觉图变化检测 contact frame
- 重建 sensor tip/base localizer
- 生成 pre-contact 样本和 Gaussian contact heatmap 标签
- 保存可视化 debug 图用于检查

Sensor localizer 训练入口：

```bash
conda activate haptic-cache
bash scripts/train_sensor_localizer.sh
```

训练结果默认写入：

- `checkpoints/sensor_localizer/`
- `outputs/metrics/sensor_localizer_metrics.json`
- `outputs/metrics/sensor_localizer_predictions.csv`
- `outputs/debug/phase1/sensor_localizer_model/`

第二阶段恢复最小预测与检索闭环：

- 训练 Tiny U-Net 或类似轻量模型预测 future contact heatmap
- 从 heatmap 中提取 Top-K contact proposal
- 复现 median error、PCK@48、bbox hit、top5 bbox hit 等指标
- 建立简单 train-cache，在验证样本上检索相似历史触觉图

## 项目结构

```text
src/          核心 Python 代码
scripts/      实验和工具入口脚本
configs/      配置文件
docs/         项目文档、论文阅读、阶段回顾
notes/        实验记录、重建记录、Codex 工作笔记
data/         本地数据集和处理后索引，不提交到 Git
outputs/      本地实验输出和 debug 图，不提交到 Git
checkpoints/  模型权重，不提交到 Git
```

## Git 注意事项

不要把大数据、视频、图片、`.npy`、`.npz`、`.pth`、`.pt`、实验输出或模型权重提交到 Git。

当前文档目录 `docs/` 保存了重要前情资料。如果它处于未追踪状态，除非明确需要整理文档提交，否则不要在普通代码或流程提交中顺手 stage 它。

## 历史最小闭环目标

数据消失前的最小闭环大致是：

```text
RGB sequence + sensor geometry
-> future contact heatmap
-> Top-K contact proposals
-> retrieve similar tactile sample from train cache
```

历史参考指标：

- median error: 约 4.0 px
- PCK@48: 约 96.8%
- bbox hit: 约 95.5%
- top5 bbox hit: 约 99.4%
