import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

from dataloader import get_dataloaders
from model import DogIdentifier

# --- HYPERPARAMETERS ---
FT_CONFIG = {
    'learning_rate': 5e-6,  
    'epochs': 10,
    'batch_size': 4, 
    'accumulation_steps': 8,
    'weight_decay': 0.01
}


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train_finetune(config_path):
    # 1. Load Task-Specific Config
    print(f"Loading config: {config_path}")
    cfg = load_config(config_path)

    DEVICE = cfg['training']['device']
    SAVE_DIR = cfg['project']['save_dir']
    TASK_NAME = cfg['project']['name']

    # Ensure save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- LOGGING SETUP ---
    log_path = f"{SAVE_DIR}/finetune_log.csv"
    history = []
    print(f"Starting Fine-Tuning Task: {TASK_NAME}")
    print(f"Logs will be saved to: {log_path}")

    # 2. Data (This automatically handles 'target_col' from config)
    train_loader, val_loader, _, num_classes, label_map = get_dataloaders(cfg)
    print(f"Detected {num_classes} classes for this task.")

    # 3. Class Weights (Handle Imbalance)
    all_labels = []
    for _, row in train_loader.dataset.data.iterrows():
        # Map the specific string label to its ID
        label_str = row[cfg['data']['target_col']]
        all_labels.append(label_map[label_str])

    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"   Weights: {class_weights}")

    # 4. Initialize Model (Fresh Head)
    model = DogIdentifier(
        num_classes=num_classes,
        model_id=cfg['model']['id'],
        freeze_encoder=False 
    ).to(DEVICE)

    # --- MULTI-GPU SUPPORT ---
    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs! Enabling DataParallel.")
        model = nn.DataParallel(model)
    else:
    # -------------------------

    # 5. SMART WEIGHT LOADING
    # We want the Encoder from Stage 1, but NOT the Classifier Head
    stage1_weights_path = "/checkpoints/best_model_w2v.pth"

    if os.path.exists(stage1_weights_path):
        print(f"Loading Stage 1 Body from: {stage1_weights_path}")
        pretrained_dict = torch.load(stage1_weights_path, map_location=DEVICE)

        # Handle DataParallel wrapping
        if isinstance(model, nn.DataParallel):
            model_dict = model.module.state_dict()
        else:
            model_dict = model.state_dict()

        # Filter out classifier weights
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.shape == model_dict[k].shape}

        # Overwrite only the matching parts
        model_dict.update(pretrained_dict)

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_dict)
        else:
            model.load_state_dict(model_dict)

        print(f"Successfully loaded {len(pretrained_dict)}/{len(model_dict)} layers (Head reset for new task).")
    else:
        print("Warning: Stage 1 weights not found. Starting from scratch (Transformers pre-trained).")

    # 6. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=FT_CONFIG['learning_rate'], weight_decay=FT_CONFIG['weight_decay'])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler('cuda')
    accumulation_steps = FT_CONFIG['accumulation_steps']

    best_val_acc = 0.0

    # 7. Training Loop
    for epoch in range(FT_CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{FT_CONFIG['epochs']}")

        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        optimizer.zero_grad()
        loop = tqdm(train_loader, desc=f"Training {TASK_NAME}")

        for i, (input_features, attention_mask, labels) in enumerate(loop):
            input_features = input_features.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.amp.autocast('cuda'):
                outputs = model(input_features, attention_mask)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            current_loss = loss.item() * accumulation_steps
            train_loss += current_loss
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            loop.set_postfix(loss=f"{current_loss:.4f}")

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # --- VALIDATION ---
        model.eval()
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for input_features, attention_mask, labels in val_loader:
                input_features = input_features.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                labels = labels.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    outputs = model(input_features, attention_mask)
                    _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val

        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Loss: {avg_train_loss:.4f}")

        # Save Logs
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        })
        pd.DataFrame(history).to_csv(log_path, index=False)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # Unwrap DataParallel before saving
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), f"{SAVE_DIR}/best_model_finetuned.pth")
            else:
                torch.save(model.state_dict(), f"{SAVE_DIR}/best_model_finetuned.pth")


if __name__ == "__main__":
    # Use argparse to select the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to the config file (e.g., config_gender.yaml)")
    args = parser.parse_args()

    train_finetune(args.config)
