import torch
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import os
import argparse
import re
from collections import Counter

# Import your modules
from src.dataloader import get_dataloaders
from src.model import DogIdentifier


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def fix_key_names(state_dict):
    """
    Translates old transformers layer names to new conventions
    to fix 'Missing key(s)' errors.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        # 1. Remove DataParallel 'module.' prefix
        k = k.replace("module.", "")

        # 2. Fix Wav2Vec2-BERT Conformer Layer Mismatches
        # Old Name -> New Name mapping based on your error log
        replacements = {
            "conv_layer_norm": "conv_module.layer_norm",
            "conv_pointwise_conv1": "conv_module.pointwise_conv1",
            "conv_depthwise_conv": "conv_module.depthwise_conv",
            "conv_depthwise_layer_norm": "conv_module.depthwise_layer_norm",
            "conv_pointwise_conv2": "conv_module.pointwise_conv2"
        }

        for old, new in replacements.items():
            if old in k:
                k = k.replace(old, new)

        new_state_dict[k] = v
    return new_state_dict


def evaluate(config_path, top_k=None):
    # 1. Setup
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file not found at {config_path}")
        return

    cfg = load_config(config_path)
    DEVICE = cfg['training']['device']
    SAVE_DIR = cfg['project']['save_dir']
    TASK_NAME = cfg['project']['name']

    print(f"🚀 Evaluating Task: {TASK_NAME}")

    # 2. Get Data
    _, _, test_loader, num_classes, label_map = get_dataloaders(cfg)

    id_to_class = {v: k for k, v in label_map.items()}
    class_names = [id_to_class[i] for i in range(num_classes)]

    # 3. Model Setup
    model = DogIdentifier(
        num_classes=num_classes,
        model_id=cfg['model']['id'],
        freeze_encoder=False
    ).to(DEVICE)

    # 4. Load Weights
    weights_path = f"{SAVE_DIR}/best_model.pth"
    # Fallback logic
    if not os.path.exists(weights_path):
        if os.path.exists(f"{SAVE_DIR}/best_model_finetuned.pth"):
            weights_path = f"{SAVE_DIR}/best_model_finetuned.pth"
        elif os.path.exists(f"{SAVE_DIR}/best_model_w2v.pth"):
            weights_path = f"{SAVE_DIR}/best_model_w2v.pth"

    if not os.path.exists(weights_path):
        print(f"❌ Error: No checkpoint found in {SAVE_DIR}")
        return

    print(f"⚖️ Loading weights: {weights_path}")

    # --- LOAD AND FIX WEIGHTS ---
    try:
        # Load raw state dict
        raw_state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        # Apply name fixes
        clean_state_dict = fix_key_names(raw_state_dict)
        # Load into model
        model.load_state_dict(clean_state_dict)
        print("✅ Weights loaded successfully (Keys patched).")
    except Exception as e:
        print(f"❌ Critical Error loading weights: {e}")
        return

    # 5. Run Inference
    model.eval()
    all_preds = []
    all_labels = []

    print("Running Inference...")
    with torch.no_grad():
        for input_features, attention_mask, labels in tqdm(test_loader):
            input_features = input_features.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.amp.autocast('cuda'):
                outputs = model(input_features, attention_mask)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 6. Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

    print("\n" + "=" * 30)
    print(f"Overall Accuracy:   {acc * 100:.2f}%")
    print(f"Macro F1-Score:     {f1:.4f}")
    print("=" * 30)

    # 7. Confusion Matrix Logic
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    final_cm = cm
    final_classes = class_names

    if top_k and num_classes > top_k:
        print(f"📉 Filtering Confusion Matrix to Top {top_k} Classes (by Frequency)...")
        label_counts = Counter(all_labels)
        top_common = label_counts.most_common(top_k)
        top_indices = [x[0] for x in top_common]
        top_indices = sorted(top_indices, key=lambda idx: natural_sort_key(class_names[idx]))

        final_cm = cm[np.ix_(top_indices, top_indices)]
        final_classes = [class_names[i] for i in top_indices]

    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = final_cm.astype('float') / (final_cm.sum(axis=1)[:, np.newaxis])
    cm_norm = np.nan_to_num(cm_norm)

    # 8. Plot
    print("Generating Plot...")
    plt.figure(figsize=(12, 10))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        xticklabels=final_classes,
        yticklabels=final_classes,
        vmax=1.0, vmin=0.0
    )
    plt.ylabel('Actual Dog')
    plt.xlabel('Predicted Dog')
    title_suffix = f"(Top {len(final_classes)} Dogs)" if top_k else "(All Classes)"
    plt.title(f'{TASK_NAME} Confusion Matrix {title_suffix}\nAcc: {acc:.2%}')

    plot_path = f"{SAVE_DIR}/confusion_matrix_top{len(final_classes)}.png"
    plt.savefig(plot_path)
    print(f"✅ Saved plot to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to config")
    parser.add_argument('--top_k', type=int, default=None, help="Plot only top K classes")
    args = parser.parse_args()

    evaluate(args.config, top_k=args.top_k)