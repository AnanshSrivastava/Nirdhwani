import soundfile as sf
import librosa
import sys

# usage: python fix_noise_samplerate.py real_noise.wav
filepath = sys.argv[1] if len(sys.argv) > 1 else "real_noise.wav"

audio, sr = sf.read(filepath)
if audio.ndim > 1:
    audio = audio.mean(axis=1)  # downmix stereo to mono

print(f"Original: {sr} Hz, {len(audio)/sr:.2f} sec")

if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sr = 16000

sf.write(filepath, audio, 16000)
print(f"Fixed:    {sr} Hz, {len(audio)/sr:.2f} sec -- saved back to {filepath}")