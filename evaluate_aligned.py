"""
Same as evaluate_batch.py, but first finds and corrects for a small timing
delay between clean and enhanced (common with block-based real-time
processing), using cross-correlation to find the true offset.
"""

import numpy as np
import soundfile as sf
import argparse
import os
import glob
from pystoi import stoi
from pesq import pesq
from scipy.signal import correlate

SR = 16000
MAX_DELAY_SEARCH = 1000  # samples, generous search window


def find_delay(clean, enhanced):
    n = min(len(clean), len(enhanced), SR * 3)  # use first 3 sec for speed
    c = clean[:n]
    e = enhanced[:n]
    corr = correlate(e, c, mode='full')
    lags = np.arange(-len(c) + 1, len(e))
    valid = (lags >= -MAX_DELAY_SEARCH) & (lags <= MAX_DELAY_SEARCH)
    best_lag = lags[valid][np.argmax(corr[valid])]
    return best_lag


def align(clean, enhanced, delay):
    if delay > 0:
        enhanced = enhanced[delay:]
    elif delay < 0:
        clean = clean[-delay:]
    n = min(len(clean), len(enhanced))
    return clean[:n], enhanced[:n]


def compute_snr(clean, enhanced):
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
    snrs, stois, pesqs, delays = [], [], [], []

    for clean_path in clean_files:
        fname = os.path.basename(clean_path)
        enh_path = os.path.join(args.enhanced_dir, fname)
        if not os.path.exists(enh_path):
            continue

        clean, _ = sf.read(clean_path)
        enhanced, _ = sf.read(enh_path)

        delay = find_delay(clean, enhanced)
        delays.append(delay)
        clean_a, enhanced_a = align(clean, enhanced, delay)

        snrs.append(compute_snr(clean_a, enhanced_a))
        stois.append(stoi(clean_a, enhanced_a, SR, extended=False))
        try:
            pesqs.append(pesq(SR, clean_a, enhanced_a, 'wb'))
        except Exception:
            pass

    print(f"Evaluated {len(snrs)} files")
    print(f"Detected delay: mean {np.mean(delays):.1f} samples ({np.mean(delays)/SR*1000:.1f} ms)\n")
    print(f"SNR:  {np.mean(snrs):.2f} dB  (target: > 15 dB)  {'PASS' if np.mean(snrs) > 15 else 'BELOW TARGET'}")
    print(f"STOI: {np.mean(stois):.3f}      (target: > 0.85) {'PASS' if np.mean(stois) > 0.85 else 'BELOW TARGET'}")
    print(f"PESQ: {np.mean(pesqs):.2f}      (target: > 2.5)  {'PASS' if np.mean(pesqs) > 2.5 else 'BELOW TARGET'}")


if __name__ == "__main__":
    main()