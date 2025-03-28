from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import librosa

# 加载本地保存的模型和处理器
model = AutoModelForSpeechSeq2Seq.from_pretrained("./distil-whisper-local")
processor = AutoProcessor.from_pretrained("./distil-whisper-local")

# 加载示例音频文件
audio_path = "./audio/Such_large_processed.wav"
audio_input, _ = librosa.load(audio_path, sr=16000)

# 处理音频输入
input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features

# 进行推理
with torch.no_grad():
    predicted_ids = model.generate(input_features)

# 解码转录结果
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print("Transcription:", transcription)
