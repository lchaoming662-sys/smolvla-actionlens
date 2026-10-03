import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy

MODEL_DIR = r"E:\VLA\models\smolvla_libero_7d"
DATA_ROOT = r"E:\VLA\data\libero_actionlens"
OUT_DIR = Path(r"E:\VLA\outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EPISODES = [834, 840, 841, 845, 847, 869, 875, 878, 910, 918]
HORIZON = 50
FPS = 10
DEVICE = "cuda"

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.cuda.reset_peak_memory_stats()

dataset = LeRobotDataset(
    "lerobot/libero",
    root=DATA_ROOT,
    episodes=EPISODES,
    delta_timestamps={"action": [i / FPS for i in range(HORIZON)]},
)

policy = SmolVLAPolicy.from_pretrained(MODEL_DIR).to(DEVICE).eval()
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    dataset_stats=dataset.meta.stats,
    preprocessor_overrides={
        "device_processor": {"device": DEVICE},
        "rename_observations_processor": {
            "rename_map": {
                "observation.images.image2": "observation.images.wrist_image",
            }
        },
    },
)

frame = dataset[0]
ground_truth = frame["action"].detach().float().cpu()
valid_mask = (~frame["action_is_pad"]).detach().cpu()
task = frame["task"]
batch = preprocess(frame)

torch.cuda.synchronize()
start = time.perf_counter()
with torch.inference_mode():
    pred = policy.predict_action_chunk(batch)
    pred = postprocess(pred)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

if isinstance(pred, dict):
    pred = pred.get("action", next(iter(pred.values())))
pred = pred[0].detach().float().cpu()
gt = ground_truth
mask = valid_mask.bool()

error = (pred - gt).abs()
masked_error = error * mask.unsqueeze(-1)
mae = masked_error.sum().item() / max(mask.sum().item() * pred.shape[-1], 1)

np.savez_compressed(
    OUT_DIR / "smolvla_zero_shot_prediction.npz",
    pred=pred.numpy(),
    ground_truth=gt.numpy(),
    valid_mask=mask.numpy(),
    task=np.array([str(task)]),
)

fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
axes = axes.flatten()
for dim in range(7):
    ax = axes[dim]
    x = np.arange(HORIZON)
    valid = mask.numpy()
    ax.plot(x[valid], gt[valid, dim].numpy(), label="ground truth", linewidth=2)
    ax.plot(x[valid], pred[valid, dim].numpy(), label="SmolVLA", linewidth=2)
    ax.set_title(f"action dim {dim}")
    ax.set_xlabel("future step")
    ax.set_ylabel("value")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
axes[7].axis("off")
axes[7].text(0.0, 0.8, "Task:\n" + str(task), fontsize=12, wrap=True)
axes[7].text(0.0, 0.45, f"MAE={mae:.6f}\nLatency={elapsed:.3f}s\nPeak VRAM={torch.cuda.max_memory_allocated()/1024**3:.2f} GB", fontsize=11)
fig.suptitle("SmolVLA zero-shot action chunk prediction", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT_DIR / "smolvla_zero_shot_prediction.png", dpi=160)
plt.close(fig)

print("task=", task)
print("pred_shape=", tuple(pred.shape))
print("gt_shape=", tuple(gt.shape))
print("valid_steps=", int(mask.sum().item()))
print("mae=", mae)
print("latency_seconds=", elapsed)
print("peak_vram_gb=", torch.cuda.max_memory_allocated()/1024**3)
print("figure=", OUT_DIR / "smolvla_zero_shot_prediction.png")
print("npz=", OUT_DIR / "smolvla_zero_shot_prediction.npz")
print("inference_ok=True")
