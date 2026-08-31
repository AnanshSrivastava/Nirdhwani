# Downloads SESA / C3GD / GISE-51 / VGGSound / LibriSpeech
"""
Downloads the core datasets used for training pairs:
  - MUSAN       (real noise: music/speech/noise categories)      -- verified link
  - LibriSpeech (real clean speech, dev-clean subset ~337MB)      -- verified link
  - SESA / GISE-51 (gunshot/explosion/siren)                      -- verify link before running

Run:  python download_datasets.py
"""

import os
import tarfile
import urllib.request

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)


def download_and_extract(url, dest_folder, archive_name):
    """Downloads a tar.gz/tgz archive and extracts it, skipping if already done."""
    archive_path = os.path.join(RAW_DIR, archive_name)
    extract_path = os.path.join(RAW_DIR, dest_folder)

    if os.path.exists(extract_path):
        print(f"[skip] {dest_folder} already exists at {extract_path}")
        return

    print(f"[downloading] {url}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
        print(f"\r  {pct}% ({downloaded // (1024*1024)}MB)", end="")

    urllib.request.urlretrieve(url, archive_path, reporthook=progress)
    print(f"\n[extracting] {archive_name}")

    with tarfile.open(archive_path) as tar:
        tar.extractall(RAW_DIR)

    os.remove(archive_path)  # free disk space once extracted
    print(f"[done] {dest_folder}\n")


# --- MUSAN: real recorded noise, music, and speech (~11GB total) ---
# Official host: OpenSLR (Johns Hopkins CLSP) — stable, long-running mirror.
download_and_extract(
    url="https://www.openslr.org/resources/17/musan.tar.gz",
    dest_folder="musan",
    archive_name="musan.tar.gz",
)

# --- LibriSpeech dev-clean: real clean speech (~337MB, ~5.4 hours) ---
# Small subset is enough for a hackathon-scale mix — swap for train-clean-100
# (~6GB) later only if you have time/disk budget to spare.
download_and_extract(
    url="https://www.openslr.org/resources/12/dev-clean.tar.gz",
    dest_folder="LibriSpeech",
    archive_name="dev-clean.tar.gz",
)

# --- SESA / GISE-51: gunshot / explosion / siren clips ---
# NOTE: verify the current hosting link before running this section.
# Both are cited across multiple papers but don't have one single stable
# canonical URL the way MUSAN/LibriSpeech do — check the dataset paper's
# GitHub/Kaggle page directly and paste the confirmed link below.
#
# download_and_extract(
#     url="<PASTE VERIFIED SESA URL HERE>",
#     dest_folder="SESA",
#     archive_name="sesa.zip",
# )

print("Core downloads complete. Check data/raw/ for folders.")
print("Remember: SESA/GISE-51 need a manually-verified link before they'll run.")