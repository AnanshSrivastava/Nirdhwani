"""
Build noisy/clean training pairs for fine-tuning DTLN, mixing clean speech
with defense noise at random SNRs.
"""

import numpy as np
import soundfile as sf
import argparse
import os
import glob
import random

SR = 16000
SNR_RANGE_DB = (-5, 15)  # covers hard (loud noise) to easy (quiet noise) cases


def load_and_resample(path, sr=SR):
    audio, file_sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    return audio


def mix_at_snr(clean, noise, snr_db):
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[:len(clean)]

    clean_power = np.mean(clean ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    noise_scaled = noise * np.sqrt(target_noise_power / noise_power)

    mixed = clean + noise_scaled
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed / peak * 0.99
        clean = clean / peak * 0.99
    return mixed, clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--noise_dir", required=True)
    parser.add_argument("--out_dir", default="./data/train_pairs")
    parser.add_argument("--n_samples", type=int, default=2000)
    args = parser.parse_args()

    clean_files = glob.glob(os.path.join(args.clean_dir, "**", "*.flac"), recursive=True)
    clean_files += glob.glob(os.path.join(args.clean_dir, "**", "*.wav"), recursive=True)
    noise_files = glob.glob(os.path.join(args.noise_dir, "**", "*.wav"), recursive=True)

    if not clean_files:
        raise FileNotFoundError(f"No clean speech files found in {args.clean_dir}")
    if not noise_files:
        raise FileNotFoundError(f"No noise files found in {args.noise_dir}")
    print(f"Found {len(clean_files)} clean files, {len(noise_files)} noise files")

    noisy_dir = os.path.join(args.out_dir, "noisy")
    target_dir = os.path.join(args.out_dir, "clean")
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    for i in range(args.n_samples):
        clean_path = random.choice(clean_files)
        noise_path = random.choice(noise_files)
        snr_db = random.uniform(*SNR_RANGE_DB)

        try:
            clean = load_and_resample(clean_path)
            noise = load_and_resample(noise_path)
        except Exception as e:
            print(f"Skipping pair due to error: {e}")
            continue

        if len(clean) < SR * 0.5:
            continue

        mixed, clean_trimmed = mix_at_snr(clean, noise, snr_db)

        fname = f"sample_{i:05d}.wav"
        sf.write(os.path.join(noisy_dir, fname), mixed, SR)
        sf.write(os.path.join(target_dir, fname), clean_trimmed, SR)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{args.n_samples} pairs created")

    print(f"\nDone. {args.n_samples} noisy/clean pairs in {args.out_dir}")


if __name__ == "__main__":
    main()