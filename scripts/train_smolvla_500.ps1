. E:\VLA\activate_lerobot.ps1

lerobot-train `
  --policy.path="E:\VLA\models\smolvla_libero_7d_migrated" `
  --dataset.repo_id="lerobot/libero" `
  --dataset.root="E:\VLA\data\libero_actionlens_20" `
  --dataset.episodes="[834,840,841,845,847,869,875,878,966,972,975,977]" `
  --rename_map='{"observation.images.image2":"observation.images.wrist_image"}' `
  --policy.device=cuda `
  --policy.dtype=bfloat16 `
  --policy.push_to_hub=false `
  --batch_size=1 `
  --accelerator.gradient_accumulation.steps=8 `
  --steps=500 `
  --save_freq=100 `
  --log_freq=10 `
  --num_workers=0 `
  --output_dir="E:\VLA\runs\smolvla_actionlens_500" `
  --job_name=smolvla_actionlens_500 `
  --wandb.enable=false
