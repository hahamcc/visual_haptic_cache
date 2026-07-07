# Visual-Haptic Cache Project

本项目用于研究视觉到触觉预测中的缓存检索方法。

## 目标

- 从视频帧中预测接触区域
- 提取运动特征，包括速度、方向、接触区域
- 建立视觉-触觉缓存库
- 在线检索缓存，减少触觉生成模型的计算开销

## 项目结构

```text
src/          核心代码
scripts/      运行脚本
configs/      配置文件
notes/        实验记录和思路
data/         数据集，不提交到 Git
outputs/      输出结果，不提交到 Git
checkpoints/  模型权重，不提交到 Git
