from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(r"E:\VLA\outputs")
BASE_NPZ = OUT_DIR / "eval500_7d_base.npz"
FT_NPZ = OUT_DIR / "eval500_ft_500.npz"

base = np.load(BASE_NPZ)
ft = np.load(FT_NPZ)

gt_all = base["ground_truth"]                  # [N, H, 7]
mask_all = base["masks"].astype(bool)          # [N, H]
pred_base_all = base["predictions"]            # [N, H, 7]
pred_ft_all = ft["predictions"]                # [N, H, 7]
episode_ids = base["episode_ids"]
sample_indices = base["sample_indices"]
H = gt_all.shape[1]
D = gt_all.shape[2]


def sample_mae(pred):
    valid = mask_all[..., None]
    err = np.abs(pred - gt_all) * valid
    return err.sum(axis=(1, 2)) / np.maximum(mask_all.sum(axis=1) * D, 1)


base_scores = sample_mae(pred_base_all)
ft_scores = sample_mae(pred_ft_all)
improvement = base_scores - ft_scores

# Pick a representative sample near the median improvement.
target = np.median(improvement)
sample_i = int(np.argmin(np.abs(improvement - target)))
print("selected_sample=", sample_i)
print("episode_id=", int(episode_ids[sample_i]))
print("sample_index=", int(sample_indices[sample_i]))
print("base_mae=", float(base_scores[sample_i]))
print("ft500_mae=", float(ft_scores[sample_i]))
print("improvement=", float(improvement[sample_i]))

x = np.arange(H)
valid = mask_all[sample_i]
gt = gt_all[sample_i]
pred_base = pred_base_all[sample_i]
pred_ft = pred_ft_all[sample_i]

fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
axes = axes.flatten()
for dim in range(D):
    ax = axes[dim]
    ax.plot(x[valid], gt[valid, dim], color="black", linewidth=2.2, label="ground truth")
    ax.plot(x[valid], pred_base[valid, dim], color="#7A7A7A", linewidth=1.8, linestyle="--", label="base")
    ax.plot(x[valid], pred_ft[valid, dim], color="#1F77B4", linewidth=1.8, label="fine-tuned 500")
    ax.set_title(f"action dim {dim}")
    ax.set_xlabel("future step")
    ax.set_ylabel("value")
    ax.grid(alpha=0.25)
    if dim == 0:
        ax.legend(fontsize=8)
axes[7].axis("off")
axes[7].text(
    0.02,
    0.86,
    "Representative test sample\n\n"
    f"episode={int(episode_ids[sample_i])}\n"
    f"local sample={int(sample_indices[sample_i])}\n"
    f"base MAE={base_scores[sample_i]:.4f}\n"
    f"500-step MAE={ft_scores[sample_i]:.4f}\n"
    f"improvement={improvement[sample_i]:.4f}",
    fontsize=11,
    va="top",
)
fig.suptitle("Base vs fine-tuned 500-step SmolVLA predictions", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT_DIR / "prediction_curves_base_vs_500.png", dpi=160)
plt.close(fig)

# Average error over all test samples and action dimensions, by future step.
base_abs = np.abs(pred_base_all - gt_all)
ft_abs = np.abs(pred_ft_all - gt_all)
valid2 = mask_all[..., None]
base_abs = np.where(valid2, base_abs, np.nan)
ft_abs = np.where(valid2, ft_abs, np.nan)
base_h = np.nanmean(base_abs, axis=(0, 2))
ft_h = np.nanmean(ft_abs, axis=(0, 2))
base_h_std = np.nanstd(base_abs, axis=(0, 2))
ft_h_std = np.nanstd(ft_abs, axis=(0, 2))

fig2, ax2 = plt.subplots(figsize=(11, 5))
ax2.plot(x, base_h, color="#7A7A7A", linewidth=2, linestyle="--", label="base")
ax2.fill_between(x, base_h - base_h_std, base_h + base_h_std, color="#7A7A7A", alpha=0.15)
ax2.plot(x, ft_h, color="#1F77B4", linewidth=2.2, label="fine-tuned 500")
ax2.fill_between(x, ft_h - ft_h_std, ft_h + ft_h_std, color="#1F77B4", alpha=0.15)
ax2.set_xlabel("future step")
ax2.set_ylabel("mean absolute error")
ax2.set_title("Error vs horizon across 80 test samples")
ax2.grid(alpha=0.25)
ax2.legend()
fig2.tight_layout()
fig2.savefig(OUT_DIR / "error_vs_horizon_base_vs_500.png", dpi=160)
plt.close(fig2)

summary = {
    "selected_sample": sample_i,
    "episode_id": int(episode_ids[sample_i]),
    "sample_index": int(sample_indices[sample_i]),
    "base_mae": float(base_scores[sample_i]),
    "ft500_mae": float(ft_scores[sample_i]),
    "improvement": float(improvement[sample_i]),
    "overall_base_mae": float(np.mean(base_scores)),
    "overall_ft500_mae": float(np.mean(ft_scores)),
    "figure_prediction": str(OUT_DIR / "prediction_curves_base_vs_500.png"),
    "figure_error_horizon": str(OUT_DIR / "error_vs_horizon_base_vs_500.png"),
}
with open(OUT_DIR / "prediction_curve_comparison_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
