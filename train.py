import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
from tqdm import tqdm
import pandas as pd
from src.dataloader import get_dataloaders
from src.model import DogIdentifier


def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train():
    # 1. Load Config
    cfg_path = "D:/dog-identification/config.yaml"
    cfg = load_config(cfg_path)

    DEVICE = cfg['training']['device']
    SAVE_DIR = cfg['project']['save_dir']
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- LOGGING SETUP ---
    # We use a separate file for Stage 1 logs to avoid confusion
    log_path = f"{SAVE_DIR}/training_log_stage1.csv"
    history = []
    print(f"Training on {DEVICE}")
    print(f"📄 Logs will be saved to: {log_path}")

    # 2. Prepare Data
    train_loader, val_loader, _, num_classes, _ = get_dataloaders(cfg)
    print(f"Identifying {num_classes} unique dogs.")

    # 3. Setup Model
    model = DogIdentifier(
        num_classes=num_classes,
        model_id=cfg['model']['id'],
        freeze_encoder=cfg['model']['freeze_encoder']
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=float(cfg['training']['learning_rate']))

    # Mixed Precision Setup
    use_amp = cfg['training']['use_mixed_precision']
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    accumulation_steps = cfg['training']['accumulation_steps']
    best_val_acc = 0.0

    # 4. Training Loop
    for epoch in range(cfg['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{cfg['training']['epochs']}")

        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        optimizer.zero_grad()

        loop = tqdm(train_loader, desc="Training")
        for i, (input_features, attention_mask, labels) in enumerate(loop):
            # Move inputs to GPU
            input_features = input_features.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward (Mixed Precision)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(input_features, attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss = loss / accumulation_steps

            # Backward
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Stats
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

                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(input_features, attention_mask)
                    _, predicted = torch.max(outputs, 1)

                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val

        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Loss: {avg_train_loss:.4f}")

        # --- SAVE LOGS ---
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        })
        pd.DataFrame(history).to_csv(log_path, index=False)

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model_w2v.pth")
            print(">>> New best model saved!")


if __name__ == "__main__":
    train()