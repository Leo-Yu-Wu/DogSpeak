import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# --- CONFIG ---
CSV_PATH = "D:/dog-identification/DogSpeak_Dataset/metadata_cleaned.csv"
SAVE_IMG_PATH = "D:/dog-identification/dataset_stats.png"


def natural_sort_key(s):
    """
    Extracts the integer ID from 'dog_12' -> 12
    Handles cases where ID might be missing or non-numeric gracefully.
    """
    matches = re.findall(r'\d+', s)
    if matches:
        return int(matches[0])
    return 0


def analyze():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find {CSV_PATH}")
        return

    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    print(f"Total Audio Clips: {len(df)}")
    print(f"Total Unique Dogs: {df['dog_id'].nunique()}")

    # Breed Stats
    print("BREED DISTRIBUTION (Clips vs Unique Dogs):")
    breed_stats = df.groupby('breed').agg(
        Clips=('filename', 'count'),
        Unique_Dogs=('dog_id', 'nunique')
    ).sort_values('Clips', ascending=False)
    print(breed_stats)

    # Gender Stats
    print("GENDER DISTRIBUTION:")
    gender_stats = df['sex'].value_counts()
    print(gender_stats)

    # Imbalance Check
    print("IMBALANCE CHECK (Clips per Dog):")
    clips_per_dog = df['dog_id'].value_counts()
    print(f"Min clips for a dog: {clips_per_dog.min()}")
    print(f"Max clips for a dog: {clips_per_dog.max()}")
    print(f"Mean clips per dog:  {clips_per_dog.mean():.2f}")

    # ---  VISUALIZATION DASHBOARD ---
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('DogSpeak Dataset Analysis', fontsize=20, weight='bold')

    # Plot A: Breed Distribution (By Audio Clips)
    sns.barplot(x=breed_stats.index, y=breed_stats['Clips'], ax=axes[0, 0], palette='viridis', hue=breed_stats.index,
                legend=False)
    axes[0, 0].set_title('Total Audio Clips per Breed', fontsize=14)
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=15)

    # Plot B: Breed Distribution (By Unique Dogs)
    sns.barplot(x=breed_stats.index, y=breed_stats['Unique_Dogs'], ax=axes[0, 1], palette='magma',
                hue=breed_stats.index, legend=False)
    axes[0, 1].set_title('Unique Dogs per Breed', fontsize=14)
    axes[0, 1].set_ylabel('Count of Dogs')
    axes[0, 1].tick_params(axis='x', rotation=15)

    # Plot C: Gender Distribution
    axes[1, 0].pie(gender_stats, labels=gender_stats.index, autopct='%1.1f%%',
                   colors=['#66b3ff', '#ff9999'], startangle=90, explode=(0.05, 0))
    axes[1, 0].set_title('Gender Distribution (Clips)', fontsize=14)

    # Plot D: Imbalance (Ranked Bar Plot of All Dogs - Sorted by ID)

    # 1. Prepare Data
    df_counts = clips_per_dog.reset_index()
    df_counts.columns = ['Dog_ID', 'Clip_Count']

    # 2. Sort by Dog ID (Numeric)
    df_counts['sort_key'] = df_counts['Dog_ID'].apply(natural_sort_key)
    df_counts = df_counts.sort_values('sort_key')

    # 3. Plot
    sns.barplot(data=df_counts, x='Dog_ID', y='Clip_Count', ax=axes[1, 1], color='teal')

    axes[1, 1].set_yscale('log')  # Log scale for Y
    axes[1, 1].set_title('Clips per Dog (Sorted by ID)', fontsize=14)
    axes[1, 1].set_xlabel('Dog IDs (Sorted 1 to 156)', fontsize=12)
    axes[1, 1].set_ylabel('Number of Audio Clips (Log Scale)', fontsize=12)

    # Clean up X-axis labels
    # Show every 10th label to keep it readable
    labels = [label.get_text() for label in axes[1, 1].get_xticklabels()]
    new_labels = []
    for i, label in enumerate(labels):
        if i % 10 == 0:
            new_labels.append(label)
        else:
            new_labels.append("")  # Hide intermediate labels

    axes[1, 1].set_xticklabels(new_labels, rotation=90)

    # Add manual ticks for Y-axis clarity
    axes[1, 1].set_yticks([1, 10, 100, 1000, 10000])
    axes[1, 1].get_yaxis().set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(SAVE_IMG_PATH)
    plt.show()


if __name__ == "__main__":
    analyze()
