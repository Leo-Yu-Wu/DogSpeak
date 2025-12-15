"""
This script scan all dog-identification dataset,
and will show the lenth distribution and recommend best clip length
"""
import torchaudio

from pathlib import Path
import numpy as np
from tqdm import tqdm


def scan_dataset(root_dir, target_sample_rate=16000):
    root_path = Path(root_dir)
    print(f"Scanning {root_path} for .wav files...")

    # 1. Find all wav files
    files = list(root_path.rglob("*.wav"))
    print(f"Found {len(files)} audio files.")

    durations = []

    print("Reading metadata (this may take 1-2 minutes)...")

    for file_path in tqdm(files):
        try:
            info = torchaudio.info(str(file_path))

            # Calculate duration in seconds (num_frames / sample_rate)
            duration_sec = info.num_frames / info.sample_rate
            durations.append(duration_sec)
        except Exception as e:
            # Skip corrupted files
            continue

    return np.array(durations)


def recommend_length(durations, target_sample_rate=16000):
    # Calculate stats
    avg = np.mean(durations)
    median = np.median(durations)
    p90 = np.percentile(durations, 90)
    p95 = np.percentile(durations, 95)
    p99 = np.percentile(durations, 99)
    max_len = np.max(durations)

    print(f"Total Clips:    {len(durations)}")
    print(f"Average Length: {avg:.2f} sec")
    print(f"Median Length:  {median:.2f} sec")
    print(f"Max Length:     {max_len:.2f} sec")
    print(f"90% of clips are shorter than: {p90:.2f} sec")
    print(f"95% of clips are shorter than: {p95:.2f} sec")
    print(f"99% of clips are shorter than: {p99:.2f} sec")

    rec_sec = p95
    rec_samples = int(rec_sec * target_sample_rate)
    print(f"95th percentile ({p95:.2f}s).")
    print(f"Target Sample Rate: {target_sample_rate} Hz")
    print(f"Recommended FIXED_LENGTH: {rec_samples}")


if __name__ == "__main__":
    DATASET_ROOT = "D:/dog-identification/DogSpeak_Dataset"
    durations = scan_dataset(DATASET_ROOT)
    recommend_length(durations)
