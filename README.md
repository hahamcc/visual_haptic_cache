# Visual-Haptic Cache Project

本项目用于研究视觉到触觉预测中的缓存检索方法。

当前仓库处于重建后的推进阶段：之前的数据、代码、模型权重和实验输出曾经丢失，但 Phase 1/2 的最小闭环已经基本恢复。当前重点不再是证明流程能跑通，而是扩大可靠数据、诊断长时序误差，并改进 cache retrieval 的局部接触匹配。

## 当前状态：2026-07-20

- 当前 C2 模型可在接触前 5、10、20、30、50、75、100 帧预测未来接触区域；输入中的四个观察帧是上下文，不是四帧预测上限。
- V4 验证集有 837 个 query：C2 Top-1 Box48 为 96.77%，Top-10 覆盖为 99.64%；远距离 `probe75/100` 的 Top-1 为 90.22%，Top-10 为 98.67%。
- 当前瓶颈是局部触觉缓存排序。正确的历史触觉条目经常在几何过滤后的候选中，但排序模型还不能稳定选中它。
- 当前路线是：冻结 C2 接触框 -> 局部缓存排序 -> cache-trust/cache-miss 判断 -> 高可信时返回缓存，低可信时交给 fallback/生成模型。
- split-0 `rec_00950–rec_00999` 是封存的 final holdout，开发期间不读取其预测结果。

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

第一阶段：数据与标签基础。当前状态：基本完成。

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

第二阶段：最小预测与检索闭环。当前状态：预测较好，检索仍是瓶颈。

- 训练 Tiny U-Net 或类似轻量模型预测 future contact heatmap
- 从 heatmap 中提取 Top-K contact proposal
- 复现 median error、PCK@48、bbox hit、top5 bbox hit 等指标
- 建立简单 train-cache，在验证样本上检索相似历史触觉图

当前阶段判断：

- 原始 296-sample rebuilt baseline：test median error 约 6.1 px，PCK@48/top5 hit@48 约 100%。
- 100-record automatic-label expanded run：530 samples，test median error 约 12 px，PCK@48 约 89.8%，top5 hit@48 约 100%。
- 这说明 contact proposal 基本可用，但扩展数据后的 top1 排名、长 time-to-contact 样本和自动标签质量还需要诊断。
- cache retrieval 目前经常能找到相似物体，但不是同一个局部接触位置，因此触觉图不一定匹配。

Contact region baseline 训练入口：

```bash
conda activate haptic-cache
bash scripts/train_contact_region.sh
```

如果只想快速检查代码链路：

```bash
bash scripts/train_contact_region.sh --epochs 2
```

训练结果默认写入：

- `checkpoints/contact_region_baseline/`
- `outputs/metrics/contact_region_baseline.json`
- `outputs/metrics/contact_region_predictions.csv`
- `outputs/metrics/contact_region_retrieval.csv`
- `outputs/debug/phase2/contact_region/`
- `outputs/debug/phase2/retrieval/`

Cache retrieval 对照入口：

```bash
bash scripts/build_cache_retrieval.sh
```

它会基于已经生成的 `contact_region_predictions.csv` 对比两种 cache key：

- `direct`: contact box 坐标、sensor tip/base、方向、probe/time-to-contact 等运动几何特征
- `hybrid`: `direct` 特征加上 `48x48` 接触框内的颜色、纹理、边缘和粗空间布局特征

对照结果默认写入：

- `outputs/metrics/contact_region_retrieval_direct.csv`
- `outputs/metrics/contact_region_retrieval_hybrid.csv`
- `outputs/metrics/contact_region_retrieval_compare.csv`
- `outputs/debug/phase2/retrieval_direct/`
- `outputs/debug/phase2/retrieval_hybrid/`

Dataset expansion audit 入口：

```bash
bash scripts/audit_dataset_expansion.sh --splits 0 --contact-sample-limit 20
```

它只读取 `/mnt/data/chi/visgel/seen/images` 和 `/mnt/data/cheng` 里的大数据/历史索引，输出小型 CSV/JSON 统计，不复制原始 RGB/touch 数据到项目目录。

输出默认写入：

- `data/processed/dataset_expansion_audit_records.csv`
- `data/processed/dataset_expansion_contact_sample.csv`
- `outputs/metrics/dataset_expansion_audit.json`

自动标签试生产入口：

```bash
bash scripts/build_expanded_region_dataset.sh --split 0 --record-start 0 --record-limit 20
```

第一版自动标签使用当前 sensor localizer 预测 tip/base，把检测到的 contact frame 上的 predicted tip 作为 future contact target，并生成 `48x48` contact box 和 heatmap 标签。原始 RGB/touch 仍然留在 `/mnt/data`，项目内只保存小型标签、heatmap 和少量 debug overlay。

输出默认写入：

- `data/processed/expanded_region_dataset/region_samples_auto.csv`
- `data/processed/expanded_region_dataset/sensor_tracks_auto.csv`
- `data/processed/expanded_region_dataset/contact_index_auto.csv`
- `data/processed/expanded_region_dataset/skipped_auto.csv`
- `data/processed/expanded_region_dataset/summary_auto.json`
- `outputs/debug/phase25/expanded_region_dataset/`

Expanded contact-region baseline 训练入口：

```bash
bash scripts/train_contact_region_expanded.sh
```

快速 smoke test：

```bash
bash scripts/train_contact_region_expanded.sh --epochs 2
```

它使用 `contact_region_expanded` 配置，输出到独立路径，不覆盖 296-sample baseline：

- `checkpoints/contact_region_expanded/`
- `outputs/metrics/contact_region_expanded.json`
- `outputs/metrics/contact_region_expanded_predictions.csv`
- `outputs/debug/phase26/contact_region_expanded/`

Expanded retrieval 对照入口：

```bash
bash scripts/build_cache_retrieval_expanded.sh
```

Phase 2 proposal/retrieval 可视化使用 `48x48` 接触区域框：

- 绿色框：真实接触区域
- 紫色框：Top1 预测接触区域
- 黄色框：其余 Top-K proposal 区域

## 下一步操作方向

优先级 1：训练 cache-trust/cache-miss predictor。

- 输入只使用线上可得的信号：缓存 ranker 最优分数、Top-2 margin、手工 key margin、几何/运动特征、probe 和输入质量。
- 训练监督来自严格 OOF cache-confidence 数据，避免用 query 自己参与训练的 cache-ranker 分数校准阈值。
- 在验证集上只选择一次保守阈值：高可信返回缓存，低可信标记 `cache miss` 并交给 fallback/生成模型。

严格 OOF cache-confidence 生成入口：

```bash
bash scripts/build_phase4d_oof_cache_confidence_v4.sh
```

优先级 2：冻结开发配方后评估 final holdout。

- 在接触模型、缓存 ranker 和 trust threshold 全部固定前，不读取 split-0 `rec_00950–rec_00999` 的预测结果。
- 最终只报告一次端到端的 contact Box48、tactile MAE、SSIM、mask IoU、cache accept/miss 覆盖率和 far 分桶表现。

优先级 3：继续提升局部缓存表征。

- 保持 C2 接触框的空间约束，不能让触觉相似度单独改变接触位置。
- 扩充同一物体不同部位、边缘/角点、转向和变速的缓存样本。
- 在 trust policy 稳定后，再考虑更强的局部视觉-触觉 embedding 或生成 fallback。

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
