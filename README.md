# SmolVLA ActionLens

SmolVLA ActionLens is an offline 7-DoF vision-language-action project built with LeRobot and LIBERO. It fine-tunes a 7-dimensional SmolVLA policy on the task:

> pick up the chocolate pudding and place it in the basket

The repository contains the 500-step checkpoint, training and evaluation scripts, all experiment figures, the measured MAE results, and the full Chinese implementation plan.

## Results

Test set: 8 held-out LIBERO episodes, 10 sampled frames per episode, 80 samples total.

| Model | MAE | Change vs base |
|---|---:|---:|
| 7D base | 0.3232 | - |
| Fine-tuned 100 steps | 0.2228 | -31.1% |
| Fine-tuned 200 steps | 0.1964 | -39.2% |
| Fine-tuned 300 steps | 0.1782 | -44.9% |
| Fine-tuned 400 steps | 0.1728 | -46.5% |
| Fine-tuned 500 steps | 0.1690 | -47.7% |

![MAE comparison](assets/figures/eval_7d_base_vs_finetuned_500.png)

![Prediction comparison](assets/figures/prediction_curves_base_vs_500.png)

![Error versus horizon](assets/figures/error_vs_horizon_base_vs_500.png)

## Repository Layout

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

## Model Weights

The final model is stored with Git LFS:

```text
weights/finetuned_500/model.safetensors
```

After cloning, run:

```powershell
git lfs install
git lfs pull
```

The model uses:

```text
Inputs:
  observation.images.image
  observation.images.wrist_image
  observation.state [8]
  language instruction

Output:
  action [50, 7]
```

For LIBERO data, the original camera key `observation.images.image2` must be renamed to `observation.images.wrist_image`.

## Quick Start

Activate the prepared environment:

```powershell
. E:\VLA\activate_lerobot.ps1
```

Prepare the 20-episode LIBERO subset:

```powershell
python scripts/prepare_libero_20.py
```

Train the 500-step model:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_smolvla_500.ps1
```

Evaluate all checkpoints:

```powershell
python scripts/evaluate_libero20_base_vs_500.py
```

Regenerate the prediction comparison:

```powershell
python scripts/plot_base_vs_500_predictions.py
```

## Experimental Setup

```text
Data: lerobot/libero
Task: task_index 29
Training episodes: 12
Test episodes: 8
Action horizon: 50
FPS: 10
Batch size: 1
Gradient accumulation: 8
Optimizer steps: 500
Device: NVIDIA RTX 4060 Laptop 8 GB
Precision: bfloat16
```

## Limitations

The reported MAE is an offline action prediction metric. It is not closed-loop LIBERO success rate and does not prove real-robot performance. The experiment uses one task and a small held-out episode set.

## License

Code and documentation are provided under Apache-2.0. The base model and dataset retain their original licenses.
