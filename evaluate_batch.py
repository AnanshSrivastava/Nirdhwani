"""
Evaluate against the PS's exact targets: SNR > 15 dB, STOI > 0.85, PESQ > 2.5
Usage: python evaluate_batch.py --clean_dir ./data/test_clean --enhanced_dir ./data/test_pretrained
"""

import numpy as np
import soundfile as sf
import argparse
import os
import glob
from pystoi import stoi
from pesq import pesq

SR = 16000

def compute_snr(clean, enhanced):
    min_len = min(len(clean), len(enhanced))
    clean, enhanced = clean[:min_len], enhanced[:min_len]
    noise = clean - enhanced
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2) + 1e-10
    return 10 * np.log10(signal_power / noise_power)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--enhanced_dir", required=True)
    args = parser.parse_args()

    clean_files = sorted(glob.glob(os.path.join(args.clean_dir, "*.wav")))
    snrs, stois, pesqs = [], [], []

    for clean_path in clean_files:
        fname = os.path.basename(clean_path)
        enh_path = os.path.join(args.enhanced_dir, fname)
        if not os.path.exists(enh_path):
            continue

        clean, _ = sf.read(clean_path)
        enhanced, _ = sf.read(enh_path)
        min_len = min(len(clean), len(enhanced))
        clean, enhanced = clean[:min_len], enhanced[:min_len]

        snrs.append(compute_snr(clean, enhanced))
        stois.append(stoi(clean, enhanced, SR, extended=False))
        try:
            pesqs.append(pesq(SR, clean, enhanced, 'wb'))
        except Exception:
            pass

    if len(snrs) == 0:
        print("No matching file pairs found — check your folder paths.")
        return

    print(f"Evaluated {len(snrs)} files\n")
    print(f"SNR:  {np.mean(snrs):.2f} dB  (target: > 15 dB)  {'PASS' if np.mean(snrs) > 15 else 'BELOW TARGET'}")
    print(f"STOI: {np.mean(stois):.3f}      (target: > 0.85) {'PASS' if np.mean(stois) > 0.85 else 'BELOW TARGET'}")
    if len(pesqs) == 0:
        print("PESQ: N/A (PESQ computation failed for all files)")
    else:
        print(f"PESQ: {np.mean(pesqs):.2f}      (target: > 2.5)  {'PASS' if np.mean(pesqs) > 2.5 else 'BELOW TARGET'}")

if __name__ == "__main__":
    main()