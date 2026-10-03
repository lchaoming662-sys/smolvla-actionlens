---
library_name: lerobot
base_model: bicmol/smolvla-libero
datasets:
- lerobot/libero
tags:
- lerobot
- smolvla
- vision-language-action
- robotics
- libero
license: apache-2.0
---

# SmolVLA ActionLens Fine-tuned 500 Steps

This is a 7-dimensional SmolVLA policy fine-tuned with LeRobot on the LIBERO task:

> pick up the chocolate pudding and place it in the basket

The checkpoint was fine-tuned from `bicmol/smolvla-libero` for 500 optimizer steps using 12 training episodes. The model predicts 50 future actions with 7 dimensions per step.

## Inputs and Outputs

```text
Inputs:
  observation.images.image
  observation.images.wrist_image
  observation.state [8]
  language instruction

Output:
  action [50, 7]
```

For LIBERO datasets, rename:

```text
observation.images.image2 -> observation.images.wrist_image
```

## Offline Evaluation

Test set: 8 held-out episodes, 10 sampled frames per episode, 80 samples total.

| Model | MAE | Change vs base |
|---|---:|---:|
| 7D base | 0.3232 | - |
| 100 steps | 0.2228 | -31.1% |
| 200 steps | 0.1964 | -39.2% |
| 300 steps | 0.1782 | -44.9% |
| 400 steps | 0.1728 | -46.5% |
| 500 steps | 0.1690 | -47.7% |

![MAE comparison](assets/figures/eval_7d_base_vs_finetuned_500.png)

## Prediction Comparison

![Prediction curves](assets/figures/prediction_curves_base_vs_500.png)

![Error versus horizon](assets/figures/error_vs_horizon_base_vs_500.png)

## Training Configuration

```text
Dataset: lerobot/libero
Task: pick up the chocolate pudding and place it in the basket
Training episodes: 12
Test episodes: 8
Batch size: 1
Gradient accumulation: 8
Effective batch size: 8
Steps: 500
Action horizon: 50
FPS: 10
Precision: bfloat16
Hardware: RTX 4060 Laptop 8 GB
```

## Usage

Install LeRobot and the SmolVLA dependencies, then load:

```python
from lerobot.policies.smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained("YOUR_HF_USERNAME/smolvla-actionlens-finetuned-500")
```

## Limitations

The reported metric is offline action MAE, not closed-loop task success rate. Results are based on one LIBERO task and a small held-out episode set.
