"""
Build a batch test set of noisy/clean pairs using DEFENSE noise specifically
(the noise type we actually fine-tuned on), so we can properly measure
whether fine-tuning helped where it matters.
"""

import numpy as np
import soundfile as sf
import glob
import os
import random

SR = 16000
SNR_DB = 5
N_TEST_SAMPLES = 30

random.seed(123)

clean_files = glob.glob("./data/LibriSpeech/dev-clean/**/*.flac", recursive=True)
noise_files = glob.glob("./data/defense_noise/**/*.wav", recursive=True)

os.makedirs("./data/test_noisy", exist_ok=True)
os.makedirs("./data/test_clean", exist_ok=True)

def load_mono(path, sr=SR):
    audio, file_sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    return audio

count = 0
for i in range(N_TEST_SAMPLES):
    clean_path = random.choice(clean_files)
    noise_path = random.choice(noise_files)

    clean = load_mono(clean_path)
    noise = load_mono(noise_path)

    if len(clean) < SR * 0.5:
        continue

    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[:len(clean)]

    clean_power = np.mean(clean ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    target_noise_power = clean_power / (10 ** (SNR_DB / 10))
    noise_scaled = noise * np.sqrt(target_noise_power / noise_power)

    mixed = clean + noise_scaled
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed / peak * 0.99
        clean = clean / peak * 0.99

    fname = f"test_{i:03d}.wav"
    sf.write(f"./data/test_noisy/{fname}", mixed, SR)
    sf.write(f"./data/test_clean/{fname}", clean, SR)
    count += 1

print(f"Created {count} held-out test pairs using defense noise at {SNR_DB} dB SNR")