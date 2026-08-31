"""
Creates (noisy, clean) training pairs by mixing real clean speech with
real recorded noise -- covering all three noise types the PS asks for:

  STATIONARY      -- steady background noise (MUSAN 'noise' category:
                      wind, hum, room tone) mixed continuously across
                      the whole clip.
  NON-STATIONARY  -- changing background noise (MUSAN 'music'/ambient,
                      or helicopter/vehicle clips) mixed continuously,
                      same as stationary but the noise itself varies
                      in character over time.
  IMPULSIVE       -- short, sudden bursts (gunshot/explosion clips from
                      SESA/GISE-51/your own extracted clips) overlaid
                      at a random timestamp INSIDE an otherwise-clean or
                      lightly-noisy speech clip -- this is what teaches
                      the model to react to a sudden spike, not just
                      steady background noise.

Output structure:
  data/pairs/clean/<id>.wav   -- the unmodified clean speech (training target)
  data/pairs/noisy/<id>.wav   -- the mixed noisy version (training input)
  data/pairs/metadata.csv     -- logs which noise/SNR/type was used per pair
                                  (useful for your results writeup + debugging)

Run:  python mix_pairs.py --num_pairs 2000
"""

import os
import csv
import random
import argparse
import numpy as np
import librosa
import soundfile as sf

SR = 16000
CLEAN_SPEECH_DIR = "data/clean/LibriSpeech"
STATIONARY_NOISE_DIR = "data/clean/musan/noise"
NONSTATIONARY_NOISE_DIR = "data/clean/musan/music"   # swap in helicopter/vehicle clips too
IMPULSIVE_NOISE_DIR = "data/clean/SESA"               # gunshot/explosion/siren clips
OUTPUT_DIR = "data/pairs"

SNR_RANGE_DB = (-5, 15)          # background noise loudness range
IMPULSE_GAIN_RANGE_DB = (3, 12)  # how much LOUDER the spike is than the speech


def list_wavs(folder, exclude_prefix=None):
    """
    Recursively finds all .wav files under `folder`. Real datasets are
    rarely flat (MUSAN nests by category, SESA nests by train/test) so
    this must search subfolders, not just the top level.

    exclude_prefix: skips filenames starting with this string -- used to
    keep SESA's negative "casual" class out of the impulsive-noise pool,
    since those clips are deliberately NOT gunshot/explosion/siren sounds.
    """
    if not os.path.exists(folder):
        return []
    import glob
    files = glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True)
    if exclude_prefix:
        files = [f for f in files if not os.path.basename(f).startswith(exclude_prefix)]
    return files


def load_random_clip(folder, target_len=None, exclude_prefix=None):
    files = list_wavs(folder, exclude_prefix=exclude_prefix)
    if not files:
        return None
    path = random.choice(files)
    audio, _ = librosa.load(path, sr=SR, mono=True)

    if target_len is not None:
        if len(audio) < target_len:
            # loop short noise clips to cover the full speech length
            reps = int(np.ceil(target_len / len(audio)))
            audio = np.tile(audio, reps)
        audio = audio[:target_len]

    return audio, path


def mix_at_snr(speech, noise, snr_db):
    """Scales noise to hit a target SNR (dB) relative to speech, then adds it."""
    speech_power = np.mean(speech ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    target_noise_power = speech_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return speech + noise * scale


def add_impulsive_burst(speech, impulse, gain_db):
    """Overlays a short noise burst at a random point inside the speech clip."""
    mixed = speech.copy()
    if len(impulse) >= len(speech):
        impulse = impulse[: len(speech) // 2]  # keep it a burst, not the whole clip

    start = random.randint(0, len(speech) - len(impulse) - 1)
    gain = 10 ** (gain_db / 20)
    mixed[start:start + len(impulse)] += impulse * gain
    return mixed, start / SR  # return timestamp too, useful for eval later


def normalize(audio, peak=0.9):
    """Prevents clipping after mixing -- scales the loudest sample to `peak`."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * peak
    return audio


def build_pairs(num_pairs):
    os.makedirs(f"{OUTPUT_DIR}/clean", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/noisy", exist_ok=True)

    speech_files = list_wavs(CLEAN_SPEECH_DIR)  # now recursive by default

    if not speech_files:
        raise RuntimeError(
            f"No clean speech found in {CLEAN_SPEECH_DIR}. "
            "Run download_datasets.py and clean_and_validate.py first."
        )

    metadata_rows = []

    for i in range(num_pairs):
        speech_path = random.choice(speech_files)
        speech, _ = librosa.load(speech_path, sr=SR, mono=True)

        noise_type = random.choice(["stationary", "nonstationary", "impulsive", "mixed"])
        snr_db = random.uniform(*SNR_RANGE_DB)

        if noise_type == "stationary":
            noise, noise_path = load_random_clip(STATIONARY_NOISE_DIR, target_len=len(speech))
            noisy = mix_at_snr(speech, noise, snr_db)
            impulse_time = None

        elif noise_type == "nonstationary":
            noise, noise_path = load_random_clip(NONSTATIONARY_NOISE_DIR, target_len=len(speech))
            noisy = mix_at_snr(speech, noise, snr_db)
            impulse_time = None

        elif noise_type == "impulsive":
            impulse, noise_path = load_random_clip(IMPULSIVE_NOISE_DIR, exclude_prefix="casual")
            gain_db = random.uniform(*IMPULSE_GAIN_RANGE_DB)
            noisy, impulse_time = add_impulsive_burst(speech, impulse, gain_db)

        else:  # "mixed" -- light background noise AND a sudden spike, most realistic case
            bg_noise, bg_path = load_random_clip(STATIONARY_NOISE_DIR, target_len=len(speech))
            noisy = mix_at_snr(speech, bg_noise, snr_db + 5)  # lighter background
            impulse, imp_path = load_random_clip(IMPULSIVE_NOISE_DIR, exclude_prefix="casual")
            gain_db = random.uniform(*IMPULSE_GAIN_RANGE_DB)
            noisy, impulse_time = add_impulsive_burst(noisy, impulse, gain_db)
            noise_path = f"{bg_path} + {imp_path}"

        noisy = normalize(noisy)
        clean = normalize(speech)

        pair_id = f"pair_{i:05d}"
        sf.write(f"{OUTPUT_DIR}/clean/{pair_id}.wav", clean, SR)
        sf.write(f"{OUTPUT_DIR}/noisy/{pair_id}.wav", noisy, SR)

        metadata_rows.append({
            "id": pair_id,
            "speech_source": speech_path,
            "noise_type": noise_type,
            "noise_source": noise_path,
            "snr_db": round(snr_db, 2),
            "impulse_timestamp_sec": impulse_time,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{num_pairs} pairs generated")

    with open(f"{OUTPUT_DIR}/metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\nDone. {num_pairs} pairs written to {OUTPUT_DIR}/")
    print(f"Metadata log: {OUTPUT_DIR}/metadata.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=2000)
    args = parser.parse_args()
    build_pairs(args.num_pairs)