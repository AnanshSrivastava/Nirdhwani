import soundfile as sf
import numpy as np
from pystoi import stoi
from pesq import pesq

def load_mono(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr

def report(name, clean, other, sr):
    n = min(len(clean), len(other))
    c, o = clean[:n], other[:n]
    noise = c - o
    snr = 10 * np.log10(np.sum(c**2) / (np.sum(noise**2) + 1e-10))
    s = stoi(c, o, sr, extended=False)
    try:
        p = pesq(sr, c, o, 'wb')
    except Exception as e:
        p = f"error: {e}"
    print(f"--- {name} ---")
    print(f"  RMS level: {np.sqrt(np.mean(o**2)):.4f} (clean RMS: {np.sqrt(np.mean(c**2)):.4f})")
    print(f"  Max abs value: {np.max(np.abs(o)):.4f}")
    print(f"  SNR:  {snr:.2f} dB")
    print(f"  STOI: {s:.3f}")
    print(f"  PESQ: {p}")
    print()

clean, sr = load_mono('clean.wav')
noisy, _ = load_mono('input_folder/noisy_test.wav')
enhanced, _ = load_mono('output_folder/noisy_test.wav')

finetuned, _ = load_mono('output_folder_finetuned/noisy_test.wav')
report("NOISY (before model)", clean, noisy, sr)
report("ENHANCED (pretrained)", clean, enhanced, sr)
report("ENHANCED (fine-tuned)", clean, finetuned, sr)