# 模型权重说明

500 步模型权重已经上传到 Hugging Face Hub：

https://huggingface.co/liming662/smolvla-actionlens-finetuned-500/blob/main/model.safetensors

下载命令：

```bash
hf download liming662/smolvla-actionlens-finetuned-500 \
  --local-dir ./models/smolvla-actionlens-finetuned-500
```

本目录保留了模型的配置、tokenizer、processor 文件和训练配置。为了避免 GitHub 大文件限制，906 MB 的 `model.safetensors` 不直接放入普通 Git 历史。
