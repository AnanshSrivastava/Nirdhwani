import soundfile as sf
import librosa
import numpy as np

audio, sr = sf.read('clean.wav')
if audio.ndim > 1:
    audio = audio.mean(axis=1)

print(f"Original: {sr} Hz, {len(audio)/sr:.2f} sec")

if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sr = 16000

sf.write('clean.wav', audio, 16000)
print(f"Fixed:    {sr} Hz, {len(audio)/sr:.2f} sec -- saved back to clean.wav")