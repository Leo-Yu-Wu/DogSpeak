import torch
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import os
import argparse
import re
from collections import Counter
from dataloader import get_dataloaders
from model import DogIdentifier


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


def evaluate(config_path, show_worst=False, sequential=False, k=20, model_type="auto"):
    # 1. Setup
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    cfg = load_config(config_path)
    DEVICE = cfg['training']['device']
    SAVE_DIR = cfg['project']['save_dir']
    TASK_NAME = cfg['project']['name']

    print(f"Evaluating Task: {TASK_NAME}")

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

    # 4. Load Weights Logic
    weights_path = ""

    if model_type == "finetuned":
        # Force load the Stage 2 (Fine-Tuned) weights
        weights_path = f"{SAVE_DIR}/best_model_finetuned.pth"
    elif model_type == "stage1":
        # Force load the Stage 1 (Frozen Encoder) weights
        weights_path = f"{SAVE_DIR}/best_model.pth"
    else:  # auto
        # Default behavior: try to find the best available model
        weights_path = f"{SAVE_DIR}/best_model.pth"  # Standard training name
        if not os.path.exists(weights_path):
            if os.path.exists(f"{SAVE_DIR}/best_model_finetuned.pth"):
                weights_path = f"{SAVE_DIR}/best_model_finetuned.pth"
            elif os.path.exists(f"{SAVE_DIR}/best_model.pth"):
                weights_path = f"{SAVE_DIR}/best_model.pth"

    if not os.path.exists(weights_path):
        print(f"Error: No checkpoint found at {weights_path}")
        print(f"(Requested type: {model_type})")
        return

    print(f"Loading weights: {weights_path}")

    # --- LOAD AND FIX WEIGHTS ---
    try:
        # Load raw state dict
        raw_state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        # Apply name fixes
        clean_state_dict = fix_key_names(raw_state_dict)
        # Load into model
        model.load_state_dict(clean_state_dict)
        print("Weights loaded successfully (Keys patched).")
    except Exception as e:
        print(f"Critical Error loading weights: {e}")
        return

    # 5. Run Inference
    model.eval()
    all_preds = []
    all_labels = []

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

    # Calculate MACRO metrics (Unweighted average across all classes)
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None,
                                                                     zero_division=0)

    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    print(f"EVALUATION RESULTS: {TASK_NAME}")
    print(f"Overall Accuracy:   {acc * 100:.2f}%")
    print(f"Macro Precision:    {macro_precision:.4f}")
    print(f"Macro Recall:       {macro_recall:.4f}")
    print(f"Macro F1-Score:     {macro_f1:.4f}")

    # 7. Confusion Matrix Logic
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    final_cm = cm
    final_classes = class_names
    title_suffix = "(All Classes)"

    if show_worst and num_classes > k:
        print(f"Identifying the {k} WORST performing classes...")
        class_f1_pairs = list(enumerate(f1))
        valid_pairs = [x for x in class_f1_pairs if support[x[0]] > 0]
        sorted_pairs = sorted(valid_pairs, key=lambda x: x[1])
        worst_indices = [x[0] for x in sorted_pairs[:k]]

        final_cm = cm[np.ix_(worst_indices, worst_indices)]
        final_classes = [class_names[i] for i in worst_indices]
        title_suffix = f"(Worst {k} Dogs by F1)"

    elif sequential and num_classes > k:
        print(f"Identifying the FIRST {k} classes (Sequential Order)...")
        all_indices = list(range(num_classes))
        sorted_indices = sorted(all_indices, key=lambda idx: natural_sort_key(class_names[idx]))
        top_indices = sorted_indices[:k]

        final_cm = cm[np.ix_(top_indices, top_indices)]
        final_classes = [class_names[i] for i in top_indices]
        title_suffix = f"(First {k} Dogs by Name)"

    elif k and num_classes > k:
        print(f"Identifying Top {k} classes by Frequency...")
        label_counts = Counter(all_labels)
        top_common = label_counts.most_common(k)
        top_indices = [x[0] for x in top_common]
        top_indices = sorted(top_indices, key=lambda idx: natural_sort_key(class_names[idx]))

        final_cm = cm[np.ix_(top_indices, top_indices)]
        final_classes = [class_names[i] for i in top_indices]
        title_suffix = f"(Top {k} Dogs by Frequency)"

    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = final_cm.astype('float') / (final_cm.sum(axis=1)[:, np.newaxis])
    cm_norm = np.nan_to_num(cm_norm)

    # 8. Plot
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
    plt.title(f'{TASK_NAME} Confusion Matrix {title_suffix}\nAcc: {acc:.2%}')

    # Save with unique name based on mode
    if show_worst:
        plot_filename = "confusion_matrix_worst.png"
    elif sequential:
        plot_filename = "confusion_matrix_first_k.png"
    else:
        plot_filename = f"confusion_matrix_top{len(final_classes)}.png"

    plot_path = f"{SAVE_DIR}/{plot_filename}"
    plt.savefig(plot_path)

    # 9. Save Metrics to Text File
    report_path = f"{SAVE_DIR}/evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Task: {TASK_NAME}\n")
        f.write(f"Weights: {weights_path}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro Precision: {macro_precision:.4f}\n")
        f.write(f"Macro Recall: {macro_recall:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write("Per-Class Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to config")
    parser.add_argument('--worst', action='store_true', help="Plot worst performing classes")
    parser.add_argument('--sequential', action='store_true', help="Plot first K classes numerically")
    parser.add_argument('--k', type=int, default=20, help="Number of classes to plot (Default: 20)")

    # New argument to choose model type
    parser.add_argument('--model_type', type=str, default='auto',
                        choices=['auto', 'stage1', 'finetuned'],
                        help="Choose 'stage1' (initial), 'finetuned' (final), or 'auto' (best available)")

    args = parser.parse_args()

    evaluate(args.config, show_worst=args.worst, sequential=args.sequential, k=args.k, model_type=args.model_type)
