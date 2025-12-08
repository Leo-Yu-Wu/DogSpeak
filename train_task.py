import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import sys
from tqdm import tqdm
import pandas as pd

from src.dataloader import get_dataloaders
from src.model import DogIdentifier


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train(config_file):
    cfg = load_config(config_file)
    DEVICE = cfg['training']['device']
    SAVE_DIR = cfg['project']['save_dir']
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- LOGGING SETUP ---
    log_path = f"{SAVE_DIR}/training_log.csv"
    history = []

    print(f"Starting Task: {cfg['project']['name']}")
    print(f"Logs will be saved to: {log_path}")

    # Data
    train_loader, val_loader, _, num_classes, label_map = get_dataloaders(cfg)
    print(f"Classes: {list(label_map.keys())}")

    # Model
    model = DogIdentifier(
        num_classes=num_classes,
        model_id=cfg['model']['id'],
        freeze_encoder=cfg['model']['freeze_encoder']
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=float(cfg['training']['learning_rate']))
    use_amp = cfg['training']['use_mixed_precision']
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    accumulation_steps = cfg['training']['accumulation_steps']

    best_val_acc = 0.0

    for epoch in range(cfg['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{cfg['training']['epochs']}")
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        optimizer.zero_grad()
        loop = tqdm(train_loader, desc="Training")

        for i, (input_features, attention_mask, labels) in enumerate(loop):
            input_features, attention_mask, labels = input_features.to(DEVICE), attention_mask.to(DEVICE), labels.to(
                DEVICE)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(input_features, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            current_loss = loss.item() * accumulation_steps
            train_loss += current_loss
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(loss=f"{current_loss:.4f}")

        train_acc = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for input_features, attention_mask, labels in val_loader:
                input_features, attention_mask, labels = input_features.to(DEVICE), attention_mask.to(
                    DEVICE), labels.to(DEVICE)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(input_features, attention_mask)
                    _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # --- SAVE LOGS ---
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        })
        pd.DataFrame(history).to_csv(log_path, index=False)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
            print(">>> Saved Best Model")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_task.py <config_file>")
    else:
        train(sys.argv[1])