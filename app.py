from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

# 选择要下载的模型版本，这里以 distil-large-v3.5 为例
model_name = "distil-whisper/distil-large-v3.5"

# 加载模型和处理器
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# 保存模型和处理器到本地目录
model.save_pretrained("./distil-whisper-local")
processor.save_pretrained("./distil-whisper-local")