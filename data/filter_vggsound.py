"""
Filters vggsound.csv down to just the defense-relevant categories,
confirmed present in your actual downloaded CSV. Run this BEFORE
extract_from_video.py so you're not wasting downloads on irrelevant
everyday sounds (dogs barking, tennis, etc.)

Run:  python filter_vggsound.py --csv "DATASET sih/VGGsound/vggsound.csv"
"""

import csv
import argparse

# Substring match, case-insensitive -- catches the full label even if it
# has extra words attached (e.g. "vehicle horn, car horn, honking")
TARGET_KEYWORDS = [
    # impulsive
    "machine gun", "cap gun", "explosion",
    # non-stationary: engine/vehicle
    "helicopter", "engine accelerating", "engine knocking",
    "engine starting", "engine idling", "vehicle horn",
    # non-stationary: sirens
    "siren",
]


def main(csv_path, out_path):
    kept = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            label = row[2].lower()
            if any(kw in label for kw in TARGET_KEYWORDS):
                kept.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(kept)

    print(f"Kept {len(kept)} rows out of relevant categories")
    print(f"Written to: {out_path}")

    # quick breakdown by label
    from collections import Counter
    counts = Counter(row[2] for row in kept)
    for label, count in counts.most_common():
        print(f"  {count:5d}  {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="DATASET sih/VGGsound/vggsound_filtered.csv")
    args = parser.parse_args()
    main(args.csv, args.out)