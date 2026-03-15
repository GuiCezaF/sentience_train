import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import logging

import config as cfg
from data import get_dataloaders
from model import MiniVGGNet


def setup_logger():
    logging.basicConfig(
        filename=str(cfg.TRAINING_LOG_PATH),
        filemode="a",
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger("train")


def run_epoch(model, loader, criterion, optimizer, scaler, device, is_train):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        bar = tqdm(loader, leave=False, ncols=90, desc="  train" if is_train else "  val")
        for images, labels in bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=cfg.DEVICE.type):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast(device_type=cfg.DEVICE.type):
                    logits = model(images)
                    loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total


def train(epochs_override=None):
    logger = setup_logger()
    train_loader, val_loader, class_weights = get_dataloaders(cfg)

    model = MiniVGGNet(cfg.NUM_CLASSES).to(cfg.DEVICE)
    class_weights = class_weights.to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    epochs = epochs_override if epochs_override is not None else cfg.EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(cfg.DEVICE.type)

    best_val_acc = 0.0
    patience_counter = 0

    print(f"Dispositivo: {cfg.DEVICE}")
    print(f"Treinando por ate {epochs} epochs (patience={cfg.PATIENCE})\n")

    epoch_bar = tqdm(range(1, epochs + 1), ncols=90, desc="Epochs")
    for epoch in epoch_bar:
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, cfg.DEVICE, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, cfg.DEVICE, is_train=False)
        scheduler.step()

        msg = f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        logger.info(msg)
        epoch_bar.set_postfix(val_acc=f"{val_acc:.4f}", val_loss=f"{val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "val_acc": val_acc, "epoch": epoch}, str(cfg.BEST_MODEL_PATH))
            tqdm.write(f"  Melhor modelo salvo (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                tqdm.write(f"\nEarly stopping na epoch {epoch}.")
                break

    print(f"\nTreinamento concluido. Melhor val_acc={best_val_acc:.4f}")
