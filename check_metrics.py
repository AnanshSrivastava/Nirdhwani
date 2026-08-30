import soundfile as sf
import numpy as np
from pystoi import stoi
from pesq import pesq

clean, sr = sf.read('clean.wav')
if clean.ndim > 1:
    clean = clean.mean(axis=1)

enhanced, enh_sr = sf.read('output_folder/noisy_test.wav')
if enhanced.ndim > 1:
    enhanced = enhanced.mean(axis=1)

print(f"clean.wav: {len(clean)} samples at {sr} Hz = {len(clean)/sr:.2f} sec")
print(f"enhanced:  {len(enhanced)} samples at {enh_sr} Hz = {len(enhanced)/enh_sr:.2f} sec")

n = min(len(clean), len(enhanced))
clean, enhanced = clean[:n], enhanced[:n]

noise = clean - enhanced
snr = 10 * np.log10(np.sum(clean**2) / (np.sum(noise**2) + 1e-10))
s = stoi(clean, enhanced, sr, extended=False)
p = pesq(sr, clean, enhanced, 'wb')

print(f'\nSNR:  {snr:.2f} dB (target >15)')
print(f'STOI: {s:.3f} (target >0.85)')
print(f'PESQ: {p:.2f} (target >2.5)')