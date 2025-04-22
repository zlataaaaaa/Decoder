import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import pandas as pd

from model import CRNN  # ваша обновлённая модель с SE и residual
from dataset import MorseDataset, collate_fn
from morse_alphabet import IDX_TO_CHAR

# === Конфиг ===
DATA_DIR    = 'data/audio/morse_dataset'
CSV_PATH    = 'data/train.csv'
CACHE_DIR   = 'cache/features'
BATCH_SIZE  = 32
LR          = 3e-4
EPOCHS      = 30
PATIENCE    = 5
MODEL_PATH  = 'checkpoints/best_model.pt'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Данные ===
df = pd.read_csv(CSV_PATH)
train_len = int(0.95 * len(df))
val_len   = len(df) - train_len
train_df, val_df = random_split(df, [train_len, val_len], generator=torch.Generator().manual_seed(42))

train_ds = MorseDataset(train_df.dataset.iloc[train_df.indices], DATA_DIR, precompute=True, cache_dir=CACHE_DIR)
val_ds   = MorseDataset(val_df.dataset.iloc[val_df.indices], DATA_DIR, precompute=True, cache_dir=CACHE_DIR)

train_dl = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate_fn, num_workers=4, pin_memory=True
)
val_dl = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_fn, num_workers=2, pin_memory=True
)

# === Модель, оптимизатор, AMP, scheduler ===
model      = CRNN(num_classes=len(IDX_TO_CHAR)).to(DEVICE)
ctc_loss   = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer  = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
# OneCycleLR с warm-up 10% от steps
scheduler  = OneCycleLR(
    optimizer,
    max_lr=LR,
    steps_per_epoch=len(train_dl),
    epochs=EPOCHS,
    pct_start=0.1,
    anneal_strategy='cos',
    cycle_momentum=False
)
scaler     = GradScaler()

# === Early Stopping ===
best_val_loss = float('inf')
epochs_no_improve = 0
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# === Тренировочный цикл ===
start_time = time.time()
for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0.0
    for features, targets, input_lengths, target_lengths in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        features, targets = features.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
        input_lens = input_lengths // 8

        optimizer.zero_grad()
        with autocast():
            logits    = model(features)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            loss      = ctc_loss(log_probs, targets, input_lens, target_lengths)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item()
    avg_train = train_loss / len(train_dl)
    print(f"Epoch {epoch} | Train Loss: {avg_train:.4f}")

    # === Валидация ===
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for features, targets, input_lengths, target_lengths in tqdm(val_dl, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            features, targets = features.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            input_lens = input_lengths // 8
            with autocast():
                logits    = model(features)
                log_probs = torch.nn.functional.log_softmax(logits, dim=2)
                loss      = ctc_loss(log_probs, targets, input_lens, target_lengths)
            val_loss += loss.item()
    avg_val = val_loss / len(val_dl)
    epoch_time = (time.time() - start_time) / 60
    print(f"Epoch {epoch} | Val Loss: {avg_val:.4f} | Time Elapsed: {epoch_time:.1f}m")


    # === Early Stopping & Save Best ===
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"--> Saved new best model (Val Loss: {avg_val:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered")
            break

total_time = (time.time() - start_time) / 60
print(f"Training completed in {total_time:.1f} minutes")
