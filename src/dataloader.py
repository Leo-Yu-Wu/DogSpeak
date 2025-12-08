import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import random
from transformers import AutoFeatureExtractor


class DogTaskDataset(Dataset):
    def __init__(self, dataframe, label_mapping, processor, target_sample_rate, max_duration_sec, target_col):
        self.data = dataframe.reset_index(drop=True)
        self.label_mapping = label_mapping
        self.processor = processor
        self.target_sample_rate = target_sample_rate
        self.target_col = target_col  # 'sex' or 'breed'

        # Calculate Max Frames
        self.target_samples = int(max_duration_sec * target_sample_rate)
        self.max_frames = int(self.target_samples / 160) + 10

    def __len__(self):
        return len(self.data)

    def _pad_or_trim_waveform(self, waveform):
        channels, num_samples = waveform.shape
        if num_samples > self.target_samples:
            return waveform[:, :self.target_samples]
        elif num_samples < self.target_samples:
            padding_needed = self.target_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
        return waveform

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            file_path = row['file_path']
            # DYNAMIC LABEL SELECTION
            label_str = row[self.target_col]

            # Load & Process (Same as before)
            waveform, sample_rate = torchaudio.load(file_path)
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = self._pad_or_trim_waveform(waveform)
            audio_array = waveform.squeeze().numpy()

            inputs = self.processor(
                audio_array,
                sampling_rate=self.target_sample_rate,
                max_length=self.max_frames,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            input_features = inputs.input_features.squeeze(0)
            attention_mask = inputs.attention_mask.squeeze(0)
            label = self.label_mapping[label_str]

            return input_features, attention_mask, label

        except Exception as e:
            print(f"Error: {self.data.iloc[idx]['file_path']} - {e}")
            new_idx = random.randint(0, len(self.data) - 1)
            return self.__getitem__(new_idx)


def get_dataloaders(cfg):
    root_path = Path(cfg['data']['root_dir'])
    df = pd.read_csv(cfg['data']['metadata_file'])

    print("Mapping files...")
    file_map = {p.name: p for p in root_path.rglob("*.wav")}
    df['file_path'] = df['filename'].map(file_map)
    df = df.dropna(subset=['file_path'])

    # Filter
    min_samples = cfg['data']['min_samples_per_dog']
    counts = df['dog_id'].value_counts()
    valid_dogs = counts[counts >= min_samples].index
    df = df[df['dog_id'].isin(valid_dogs)]

    # 1. SPLIT STRATEGY: GROUPED SPLIT
    print("Performing GROUPED split...")

    splitter_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg['project']['seed'])
    train_idx, test_idx = next(splitter_test.split(df, groups=df['dog_id']))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # Further split Train into Train/Val
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=cfg['project']['seed'])
    train_sub_idx, val_sub_idx = next(splitter_val.split(train_df, groups=train_df['dog_id']))

    val_df = train_df.iloc[val_sub_idx]
    train_df = train_df.iloc[train_sub_idx]

    print(f"Split Results: Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 2. Setup Labels based on Target Column
    target_col = cfg['data']['target_col']
    unique_labels = sorted(df[target_col].unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    print(f"Task: {target_col} Classification | Classes: {unique_labels}")

    model_id = cfg['model']['id']
    processor = AutoFeatureExtractor.from_pretrained(model_id)

    sr = cfg['data']['target_sample_rate']
    dur = cfg['data']['max_duration_sec']

    # Create Datasets
    train_ds = DogTaskDataset(train_df, label_map, processor, sr, dur, target_col)
    val_ds = DogTaskDataset(val_df, label_map, processor, sr, dur, target_col)
    test_ds = DogTaskDataset(test_df, label_map, processor, sr, dur, target_col)

    bs = cfg['training']['batch_size']
    nw = cfg['data']['num_workers']

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    return train_loader, val_loader, test_loader, len(unique_labels), label_map