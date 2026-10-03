# SmolVLA ActionLens

SmolVLA ActionLens 是一个基于 LeRobot 和 LIBERO 的 7 维视觉-语言-动作离线项目。项目对 SmolVLA 进行微调，使其能够在以下任务中预测机器人未来 50 步、每步 7 维的动作序列：

> 拿起巧克力布丁并放入篮子

仓库包含 500 步微调权重、训练与评估脚本、全部实验图表、离线 MAE 结果，以及完整中文实施方案。

## 在线资源

- GitHub 仓库：https://github.com/lchaoming662-sys/smolvla-actionlens
- Hugging Face 模型：https://huggingface.co/liming662/smolvla-actionlens-finetuned-500
- 模型权重：https://huggingface.co/liming662/smolvla-actionlens-finetuned-500/blob/main/model.safetensors

## 实验结果

测试集为 8 个 LIBERO 测试 episode，每个 episode 抽取 10 帧，共 80 个测试样本。

| 模型 | MAE | 相对基础模型 |
|---|---:|---:|
| 7D 基础模型 | 0.3232 | - |
| 微调 100 步 | 0.2228 | 降低 31.1% |
| 微调 200 步 | 0.1964 | 降低 39.2% |
| 微调 300 步 | 0.1782 | 降低 44.9% |
| 微调 400 步 | 0.1728 | 降低 46.5% |
| 微调 500 步 | 0.1690 | 降低 47.7% |

![MAE 对比](assets/figures/eval_7d_base_vs_finetuned_500.png)

![预测曲线对比](assets/figures/prediction_curves_base_vs_500.png)

![误差随预测步数变化](assets/figures/error_vs_horizon_base_vs_500.png)

## 仓库结构

```text
smolvla-actionlens/
├── README.md
├── MODEL_CARD.md
├── LICENSE
├── docs/
│   └── SmolVLA_ActionLens_实施方案.docx
├── assets/figures/
├── results/
├── scripts/
└── weights/
    └── finetuned_500/
```

## 模型权重

500 步模型权重已上传到 Hugging Face Hub：

https://huggingface.co/liming662/smolvla-actionlens-finetuned-500

下载命令：

```bash
hf download liming662/smolvla-actionlens-finetuned-500 \
  --local-dir ./models/smolvla-actionlens-finetuned-500
```

模型输入和输出：

```text
输入：
  observation.images.image
  observation.images.wrist_image
  observation.state [8]
  自然语言任务指令

输出：
  action [50, 7]
```

LIBERO 数据集中的相机键需要重命名：

```text
observation.images.image2 -> observation.images.wrist_image
```

## 快速开始

激活本机准备好的 LeRobot 环境：

```powershell
. E:\VLA\activate_lerobot.ps1
```

准备 20 个 episode 的 LIBERO 子集：

```powershell
python scripts/prepare_libero_20.py
```

训练 500 步模型：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_smolvla_500.ps1
```

评估基础模型与 100 到 500 步 checkpoint：

```powershell
python scripts/evaluate_libero20_base_vs_500.py
```

生成 Base 与 500 步模型的预测曲线对比图：

```powershell
python scripts/plot_base_vs_500_predictions.py
```

## 实验配置

```text
数据集：lerobot/libero
任务：task_index 29
训练 episode：12 个
测试 episode：8 个
动作预测长度：50 步
数据帧率：10 FPS
Batch size：1
梯度累积：8
优化步数：500
设备：NVIDIA RTX 4060 Laptop 8 GB
精度：bfloat16
```

## 局限说明

当前结果是在 8 个测试 episode、80 个样本上的离线动作预测 MAE，不代表 LIBERO 闭环任务成功率，也不代表真实机器人性能。实验只覆盖单一抓放任务和较小的 episode 子集。

## 许可证

代码和文档采用 Apache-2.0 许可证。基础模型和数据集遵循其原始许可证。
