import gc
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy

DATA_ROOT = r"E:\VLA\data\libero_actionlens_20"
TEST_EPISODES = [910, 918, 920, 929, 944, 948, 949, 953]
HORIZON = 50
FPS = 10
DEVICE = "cuda"
SAMPLES_PER_EPISODE = 10
OUT_DIR = Path(r"E:\VLA\outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RENAME_MAP = {"observation.images.image2": "observation.images.wrist_image"}

MODELS = {
    "7d_base": r"E:\VLA\models\smolvla_libero_7d_migrated",
    "ft_100": r"E:\VLA\runs\smolvla_actionlens_500\checkpoints\000100\pretrained_model",
    "ft_200": r"E:\VLA\runs\smolvla_actionlens_500\checkpoints\000200\pretrained_model",
    "ft_300": r"E:\VLA\runs\smolvla_actionlens_500\checkpoints\000300\pretrained_model",
    "ft_400": r"E:\VLA\runs\smolvla_actionlens_500\checkpoints\000400\pretrained_model",
    "ft_500": r"E:\VLA\runs\smolvla_actionlens_500\checkpoints\000500\pretrained_model",
}

dataset = LeRobotDataset(
    "lerobot/libero",
    root=DATA_ROOT,
    episodes=TEST_EPISODES,
    delta_timestamps={"action": [i / FPS for i in range(HORIZON)]},
)
meta = LeRobotDatasetMetadata("lerobot/libero")

# The selected dataset concatenates requested episodes in TEST_EPISODES order.
local_start = 0
sample_indices = []
lengths = []
for ep in TEST_EPISODES:
    length = int(meta.episodes[ep]["length"])
    lengths.append(length)
    max_start = max(1, length - HORIZON)
    for frac in np.linspace(0.05, 0.95, SAMPLES_PER_EPISODE):
        sample_indices.append(local_start + int(round(frac * max_start)))
    local_start += length

assert local_start == len(dataset), (local_start, len(dataset))
frames = [dataset[i] for i in sample_indices]
print("test_episodes=", TEST_EPISODES)
print("episode_lengths=", lengths)
print("test_samples=", len(frames), "dataset_frames=", len(dataset))


def evaluate_model(name, model_dir):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    policy = SmolVLAPolicy.from_pretrained(model_dir).to(DEVICE).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_dir,
        preprocessor_overrides={
            "device_processor": {"device": DEVICE},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )

    errors = []
    predictions = []
    ground_truths = []
    masks = []
    latencies = []
    episode_ids = []

    for frame in frames:
        gt = frame["action"].detach().float().cpu()
        mask = (~frame["action_is_pad"]).detach().cpu().bool()
        batch = preprocess(frame)

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            pred = policy.predict_action_chunk(batch)
            pred = postprocess(pred)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)

        if isinstance(pred, dict):
            pred = pred.get("action", next(iter(pred.values())))
        pred = pred[0].detach().float().cpu()
        err = (pred - gt).abs() * mask.unsqueeze(-1)
        mae = err.sum().item() / max(mask.sum().item() * pred.shape[-1], 1)
        errors.append(mae)
        predictions.append(pred.numpy())
        ground_truths.append(gt.numpy())
        masks.append(mask.numpy())
        episode_ids.append(int(frame["episode_index"].item()))

    result = {
        "model": name,
        "model_dir": model_dir,
        "test_episodes": TEST_EPISODES,
        "samples": len(errors),
        "mae_mean": float(np.mean(errors)),
        "mae_std": float(np.std(errors)),
        "mae_per_sample": [float(x) for x in errors],
        "latency_mean_seconds": float(np.mean(latencies)),
        "latency_std_seconds": float(np.std(latencies)),
        "peak_vram_gb": float(torch.cuda.max_memory_allocated() / 1024**3),
    }

    np.savez_compressed(
        OUT_DIR / f"eval500_{name}.npz",
        predictions=np.stack(predictions),
        ground_truth=np.stack(ground_truths),
        masks=np.stack(masks),
        sample_indices=np.array(sample_indices),
        episode_ids=np.array(episode_ids),
    )

    del policy, preprocess, postprocess
    gc.collect()
    torch.cuda.empty_cache()
    return result


results = []
for name, model_dir in MODELS.items():
    print("evaluating=", name)
    results.append(evaluate_model(name, model_dir))

with open(OUT_DIR / "eval_7d_base_vs_finetuned_500.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

labels = [r["model"] for r in results]
means = [r["mae_mean"] for r in results]
stds = [r["mae_std"] for r in results]
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(labels, means, yerr=stds, capsize=6, color=["#6B7280", "#8AB6D6", "#5B9BD5", "#2E75B6", "#1F4E79", "#153E5C"])
ax.set_ylabel("MAE")
ax.set_title("LIBERO 8-episode test set: base vs fine-tuned checkpoints")
ax.grid(axis="y", alpha=0.25)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.005, f"{m:.4f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "eval_7d_base_vs_finetuned_500.png", dpi=160)
plt.close(fig)

print(json.dumps(results, ensure_ascii=False, indent=2))
print("metrics=", OUT_DIR / "eval_7d_base_vs_finetuned_500.json")
print("figure=", OUT_DIR / "eval_7d_base_vs_finetuned_500.png")
print("evaluation_ok=True")
