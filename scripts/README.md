# 脚本说明

- `prepare_libero_20.py`：构建 20 个 episode 的训练和测试子集。
- `load_libero_subset.py`：验证前 10 个 episode 的数据格式。
- `smolvla_zero_shot_inference.py`：执行一次零样本动作预测并保存曲线。
- `train_smolvla_500.ps1`：训练 500 步 SmolVLA 模型。
- `evaluate_libero20_base_vs_500.py`：在 8 个测试 episode 上评估基础模型和 100 到 500 步 checkpoint。
- `plot_base_vs_500_predictions.py`：生成 Base 与 500 步模型的预测对比图和误差曲线。
- `save_libero_sample_preview.py`：导出 LIBERO 双相机样本预览图。
