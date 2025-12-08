"""
Clean DogSpeak Dataset: remove any broken files from the metadata.csv file
create a matadata_cleaned.csv file
"""
import pandas as pd
import torchaudio
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
DATASET_ROOT = "D:/dog-identification/DogSpeak_Dataset"
INPUT_CSV = f"{DATASET_ROOT}/metadata.csv"
OUTPUT_CSV = f"{DATASET_ROOT}/metadata_cleaned.csv"
MIN_SAMPLES = 10


def clean_dataset():
    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    original_len = len(df)

    # 1. Map paths and check existence
    root_path = Path(DATASET_ROOT)
    print("Mapping files on disk...")
    files_on_disk = {p.name: p for p in root_path.rglob("*.wav")}

    df['file_path'] = df['filename'].map(files_on_disk)

    # Drop rows where file doesn't exist
    df = df.dropna(subset=['file_path'])
    print(f"Found {len(df)} files matching metadata.")

    # 2. Filter by Minimum Samples
    print(f"\nFiltering dogs with < {MIN_SAMPLES} clips...")
    counts = df['dog_id'].value_counts()
    valid_dogs = counts[counts >= MIN_SAMPLES].index

    # Identify dropped dogs
    dropped_dogs = counts[counts < MIN_SAMPLES]
    print(f"Dropping {len(dropped_dogs)} dogs (Total {dropped_dogs.sum()} clips).")

    # Keep only valid dogs
    df_clean = df[df['dog_id'].isin(valid_dogs)].copy()

    # 3. Scan for Corrupt Files (The 'soundfile' error fix)
    print("\nScanning for corrupt audio files (this takes a few minutes)...")
    valid_indices = []
    corrupt_count = 0

    for idx, row in tqdm(df_clean.iterrows(), total=len(df_clean)):
        try:
            # load the header info
            # If the header is broken, torchaudio.info usually fails
            torchaudio.info(str(row['file_path']))
            valid_indices.append(idx)
        except Exception as e:
            print(f"Corrupt file found: {row['filename']} - Removing!")
            corrupt_count += 1

    # Apply the clean filter
    df_final = df_clean.loc[valid_indices]

    # 4. Save
    print("-" * 30)
    print(f"Original Clips: {original_len}")
    print(f"Cleaned Clips:  {len(df_final)}")
    print(f"Removed:        {original_len - len(df_final)} ({corrupt_count} were corrupt)")
    print("-" * 30)

    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved cleaned metadata to: {OUTPUT_CSV}")
    print("Update your config.yaml to use this new file!")


if __name__ == "__main__":
    clean_dataset()