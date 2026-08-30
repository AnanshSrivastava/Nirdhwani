import numpy as np
import soundfile as sf

SR = 16000
SNR_DB = 5

clean, sr = sf.read("clean.wav")
if clean.ndim > 1:
    clean = clean.mean(axis=1)

noise, nsr = sf.read("real_noise.wav")
if noise.ndim > 1:
    noise = noise.mean(axis=1)

if len(noise) < len(clean):
    reps = int(np.ceil(len(clean) / len(noise)))
    noise = np.tile(noise, reps)
noise = noise[:len(clean)]

clean_power = np.mean(clean ** 2) + 1e-10
noise_power = np.mean(noise ** 2) + 1e-10
target_noise_power = clean_power / (10 ** (SNR_DB / 10))
noise_scaled = noise * np.sqrt(target_noise_power / noise_power)

noisy = clean + noise_scaled
peak = np.max(np.abs(noisy))
if peak > 0.99:
    noisy = noisy / peak * 0.99

sf.write("input_folder/noisy_test.wav", noisy, SR)
print(f"Created input_folder/noisy_test.wav at {SNR_DB} dB SNR using real noise")