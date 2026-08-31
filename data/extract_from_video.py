"""
Extracts real audio clips from YouTube videos for your custom dataset.

Two ways to use this:

  MODE 1 -- from a CSV (e.g. VGGSound's metadata, or your own list)
            Columns needed: youtube_id, start_seconds, label
            Downloads just the relevant ~10s window per row -- not the
            whole video -- and saves it as a labeled clip.

  MODE 2 -- from a list of full YouTube URLs you found yourself
            (training footage, range demos, documentaries). Downloads
            the full audio, then either:
              (a) auto-detects likely impulsive events (sudden volume
                  spikes -- candidate gunshots/explosions) so you don't
                  have to scrub the whole video by ear, or
              (b) you mark timestamps yourself in Audacity and feed them
                  back in via a small CSV (see manual_timestamps.csv
                  format below).

Requires: pip install yt-dlp librosa soundfile numpy

Run:
  python extract_from_video.py --mode vggsound --csv vggsound.csv --limit 200
  python extract_from_video.py --mode url --url "https://youtube.com/watch?v=XXXX" --auto
  python extract_from_video.py --mode url --url "https://youtube.com/watch?v=XXXX" --timestamps manual_timestamps.csv
"""

import os
import csv
import sys
import argparse
import subprocess
import numpy as np
import librosa
import soundfile as sf

SR = 16000
OUTPUT_DIR = "data/clean/custom_extracted"
TEMP_DIR = "data/raw/video_audio_temp"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


def download_audio(youtube_id_or_url, out_path):
    """Downloads audio-only from a YouTube video using yt-dlp."""
    url = youtube_id_or_url if youtube_id_or_url.startswith("http") \
        else f"https://youtube.com/watch?v={youtube_id_or_url}"

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-x", "--audio-format", "wav",
        "-o", out_path.replace(".wav", ".%(ext)s"),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [failed] {url}: {result.stderr.strip()[-200:]}")
        return False
    return True


def clip_segment(audio_path, start_sec, duration_sec, out_path):
    """Cuts a short window out of a longer downloaded audio file."""
    audio, sr = librosa.load(audio_path, sr=SR, mono=True)
    start_sample = int(start_sec * SR)
    end_sample = int((start_sec + duration_sec) * SR)
    segment = audio[start_sample:end_sample]
    if len(segment) < duration_sec * SR * 0.5:
        return False  # too short (near end of video) -- skip
    sf.write(out_path, segment, SR)
    return True


def detect_spikes(audio, sr, threshold_factor=4.0, min_gap_sec=1.0):
    """
    Simple energy-based spike detector -- flags moments where volume
    jumps well above the local background level. Good for surfacing
    CANDIDATE gunshot/explosion moments in a long video so you don't
    have to scrub the whole thing by ear. Always spot-check the results;
    this catches loud transients generally, not specifically gunshots.
    """
    frame_len = int(0.05 * sr)  # 50ms frames
    energy = np.array([
        np.sqrt(np.mean(audio[i:i + frame_len] ** 2))
        for i in range(0, len(audio) - frame_len, frame_len)
    ])
    background = np.median(energy) + 1e-6
    spike_frames = np.where(energy > background * threshold_factor)[0]

    # merge nearby detections into single events
    spike_times = []
    last_time = -min_gap_sec
    for f in spike_frames:
        t = f * frame_len / sr
        if t - last_time >= min_gap_sec:
            spike_times.append(t)
            last_time = t

    return spike_times


def mode_vggsound(csv_path, limit):
    """Downloads and clips VGGSound entries from its metadata CSV."""
    with open(csv_path) as f:
        reader = csv.reader(f)
        rows = list(reader)[:limit]

    print(f"Processing {len(rows)} VGGSound entries...")
    for i, row in enumerate(rows):
        youtube_id, start_sec, label = row[0], float(row[1]), row[2]
        temp_path = os.path.join(TEMP_DIR, f"{youtube_id}.wav")

        if not os.path.exists(temp_path):
            if not download_audio(youtube_id, temp_path):
                continue

        safe_label = label.replace(" ", "_").replace("/", "-")
        out_path = os.path.join(OUTPUT_DIR, f"vgg_{safe_label}_{i:04d}.wav")
        clip_segment(temp_path, start_sec, duration_sec=10.0, out_path=out_path)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)} done")


def mode_url_auto(url):
    """Downloads a video and auto-flags candidate impulsive-noise moments."""
    temp_path = os.path.join(TEMP_DIR, "video_audio.wav")
    if not download_audio(url, temp_path):
        return

    audio, sr = librosa.load(temp_path, sr=SR, mono=True)
    spikes = detect_spikes(audio, sr)
    print(f"Found {len(spikes)} candidate spike(s) at: {[round(t, 1) for t in spikes]}")
    print("Listen to each before trusting it -- this flags loud transients generally.")

    for i, t in enumerate(spikes):
        start = max(0, t - 0.5)  # grab half a second before the spike too
        out_path = os.path.join(OUTPUT_DIR, f"auto_spike_{i:03d}.wav")
        clip_segment(temp_path, start, duration_sec=2.0, out_path=out_path)
        print(f"  saved: {out_path}")


def mode_url_manual(url, timestamps_csv):
    """
    Downloads a video and clips it at timestamps YOU marked in Audacity.

    manual_timestamps.csv format (no header):
      12.4,gunshot
      45.0,siren
      88.2,explosion
    """
    temp_path = os.path.join(TEMP_DIR, "video_audio.wav")
    if not download_audio(url, temp_path):
        return

    with open(timestamps_csv) as f:
        reader = csv.reader(f)
        rows = list(reader)

    for i, (start_sec, label) in enumerate(rows):
        out_path = os.path.join(OUTPUT_DIR, f"manual_{label}_{i:03d}.wav")
        clip_segment(temp_path, float(start_sec), duration_sec=2.0, out_path=out_path)
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vggsound", "url"], required=True)
    parser.add_argument("--csv", help="VGGSound metadata CSV path")
    parser.add_argument("--limit", type=int, default=200, help="max VGGSound rows to pull")
    parser.add_argument("--url", help="YouTube URL for manual/auto extraction")
    parser.add_argument("--auto", action="store_true", help="auto-detect spike moments")
    parser.add_argument("--timestamps", help="CSV of manually marked timestamps")
    args = parser.parse_args()

    if args.mode == "vggsound":
        mode_vggsound(args.csv, args.limit)
    elif args.mode == "url":
        if args.auto:
            mode_url_auto(args.url)
        elif args.timestamps:
            mode_url_manual(args.url, args.timestamps)
        else:
            print("For --mode url, pass either --auto or --timestamps <file>")