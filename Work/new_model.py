import os
import glob
import math
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

# --- 1. Configuration & Seeds ---
MODEL_NAME = "xlm-roberta-large" # Or "xlm-roberta-large" if memory allows
# Adjust these paths to your Kaggle dataset name
BASE_DIR = "/kaggle/input/semeval/subtask1"
TRAIN_DIR = os.path.join(BASE_DIR, "train/")
DEV_DIR   = os.path.join(BASE_DIR, "dev/")

MAX_LEN = 256
BATCH_SIZE = 16
GRAD_ACCUM = 2
EPOCHS = 6
LR = 3e-5
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Folder-Based Data Loading (Mirroring DeBERTa Notebook) ---
def load_language_data(directory):
    files = glob.glob(os.path.join(directory, "*.csv"))
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        # Extract language code from filename (e.g., 'urd.csv' -> 'urd')
        lang = os.path.basename(file).split(".")[0]
        df["language"] = lang
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

train_full = load_language_data(TRAIN_DIR)
test_df = load_language_data(DEV_DIR)

# Stratified Split by Polarization AND Language
train_df, val_df = train_test_split(
    train_full,
    test_size=0.15,
    stratify=train_full[["polarization", "language"]],
    random_state=SEED
)

print(f"Loaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples.")

# --- 3. Tokenization & Datasets ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

# Convert to HF Datasets for faster mapping
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tokenize_fn, batched=True)
val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True)).map(tokenize_fn, batched=True)
test_ds  = Dataset.from_pandas(test_df.reset_index(drop=True)).map(tokenize_fn, batched=True)

# Keep only necessary columns for the loaders
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "polarization"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "polarization"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

data_collator = DataCollatorWithPadding(tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE*2, collate_fn=data_collator)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE*2, collate_fn=data_collator)

# --- 4. Class Weights & Model ---
classes = np.unique(train_df["polarization"].values)
weights = compute_class_weight("balanced", classes=classes, y=train_df["polarization"].values)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(classes)
).to(device)

# --- 5. Optimizer, Scheduler & AMP ---
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps
)

# Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# --- 6. Training Loop with Per-Language Logic ---
@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "polarization"}
        labels = batch["polarization"].to(device)
        logits = model(**inputs).logits
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    return f1, acc, np.array(all_preds), np.array(all_labels)

best_f1 = 0
output_path = "/kaggle/working/best_model"

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for step, batch in enumerate(pbar):
        labels = batch.pop("polarization").to(device)
        inputs = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast(): # AMP for speed
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels) / GRAD_ACCUM

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        running_loss += loss.item() * GRAD_ACCUM
        pbar.set_postfix({"loss": running_loss / (step + 1)})

    # Validation
    val_f1, val_acc, _, _ = evaluate_model(model, val_loader)
    print(f"Val Macro F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print("💾 Best model saved!")

# --- 7. Final Language-Wise Metrics ---
# Reload best and test
best_model = AutoModelForSequenceClassification.from_pretrained(output_path).to(device)
_, _, test_preds, test_labels = evaluate_model(best_model, val_loader) # Use val as dev set reference

# Per-language F1 score calculation
def get_per_lang_f1(labels, preds, languages):
    scores = {}
    for lang in np.unique(languages):
        idx = np.where(languages == lang)
        scores[lang] = f1_score(labels[idx], preds[idx], average="macro")
    return scores

val_langs = val_df["language"].values
lang_scores = get_per_lang_f1(test_labels, test_preds, val_langs)
print("Per-language F1 on Validation:", lang_scores)
