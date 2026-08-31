"""
Cleans and standardizes all raw audio before mixing.

Why this step matters: your raw sources (MUSAN, LibriSpeech, SESA, your own
video-extracted clips) will come in different sample rates, channel counts,
and occasionally contain corrupted or near-silent files. Training on
inconsistent audio quietly wrecks your model's quality without an obvious
error message -- this step catches it up front.

What it does, per file:
  1. Load the file (skip and log if it can't be read -- corrupted file)
  2. Resample to 16kHz mono (the standard rate for speech-enhancement models,
     matches DTLN's expected input)
  3. Drop it if it's shorter than MIN_DURATION or near-silent (likely a bad clip)
  4. Save the cleaned version to data/clean/<same relative path>

Run:  python clean_and_validate.py
"""

import os
import soundfile as sf
import librosa
import numpy as np

RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
TARGET_SR = 16000
MIN_DURATION_SEC = 0.3       # drop anything shorter -- too short to be useful
SILENCE_RMS_THRESHOLD = 1e-4  # drop near-silent clips (likely broken recordings)

os.makedirs(CLEAN_DIR, exist_ok=True)

stats = {"processed": 0, "dropped_corrupt": 0, "dropped_short": 0, "dropped_silent": 0}


def clean_file(src_path, dst_path):
    try:
        audio, sr = librosa.load(src_path, sr=None, mono=True)
    except Exception as e:
        print(f"[corrupt] {src_path}: {e}")
        stats["dropped_corrupt"] += 1
        return

    duration = len(audio) / sr
    if duration < MIN_DURATION_SEC:
        stats["dropped_short"] += 1
        return

    rms = np.sqrt(np.mean(audio ** 2))
    if rms < SILENCE_RMS_THRESHOLD:
        stats["dropped_silent"] += 1
        return

    # Resample to the standard 16kHz mono target if needed
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    sf.write(dst_path, audio, TARGET_SR, subtype="PCM_16")
    stats["processed"] += 1


def walk_and_clean(raw_subdir):
    """Walks a raw dataset folder and cleans every .wav/.flac/.mp3 file found."""
    src_root = os.path.join(RAW_DIR, raw_subdir)
    if not os.path.exists(src_root):
        print(f"[skip] {src_root} not found -- did you run download_datasets.py?")
        return

    for dirpath, _, filenames in os.walk(src_root):
        for fname in filenames:
            if not fname.lower().endswith((".wav", ".flac", ".mp3")):
                continue
            src_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(src_path, RAW_DIR)
            dst_path = os.path.join(CLEAN_DIR, os.path.splitext(rel_path)[0] + ".wav")
            clean_file(src_path, dst_path)


if __name__ == "__main__":
    for subdir in ["musan", "LibriSpeech", "SESA", "custom_extracted"]:
        print(f"\n[cleaning] {subdir}")
        walk_and_clean(subdir)

    print("\n--- Summary ---")
    print(f"Processed & kept: {stats['processed']}")
    print(f"Dropped (corrupt): {stats['dropped_corrupt']}")
    print(f"Dropped (too short): {stats['dropped_short']}")
    print(f"Dropped (near-silent): {stats['dropped_silent']}")
    print(f"\nCleaned files are in: {CLEAN_DIR}/")