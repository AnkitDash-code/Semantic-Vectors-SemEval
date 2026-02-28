# ============================================================
# SemEval 2026 Task 9 - Subtask 1
# XLM-RoBERTa LARGE + Improved Architecture
# Training + Epoch-wise Macro-F1 + Threshold Tuning + Prediction
# ============================================================

# ---------- IMPORTS ----------
import os, glob, random
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

# ---------- CONFIG ----------
# Mirrored from new_model.py (adjusted to keep variable names used in this file)
MODEL_NAME = "xlm-roberta-large"
BASE_DIR = "/kaggle/input/semeval/subtask1"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
DEV_DIR = os.path.join(BASE_DIR, "dev")
PRED_DIR = os.path.join(BASE_DIR, "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

# Keep encoder name used elsewhere in the file
ENCODER_MODEL = MODEL_NAME

# Device as torch.device to use with .to(DEVICE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 256
BATCH_SIZE = 16
# Gradient accumulation (kept name ACCUM_STEPS used in training loop)
ACCUM_STEPS = 2
EPOCHS = 6
LR = 3e-05
TEMPERATURE = 0.07
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ---------- LANGUAGE CLUSTERS ----------
LANG_CLUSTERS = {
    "western": ["eng", "deu", "ita", "spa", "pol"],
    "indic": ["hin", "ben", "tel", "ori", "pan", "urd", "nep"],
    "semitic": ["arb", "fas"],
    "african": ["amh", "hau", "swa"],
    "southeast_asia": ["mya", "khm"],
    "sinitic": ["zho"],
    "turkic": ["tur"],
    "slavic": ["rus"],
}

LANG2CLUSTER = {l: c for c, langs in LANG_CLUSTERS.items() for l in langs}
CLUSTER2ID = {c: i for i, c in enumerate(LANG_CLUSTERS)}

# ---------- DATA ----------
def load_train_data():
    files = glob.glob(f"{TRAIN_DIR}/*.csv")
    dfs = []
    for f in files:
        lang = os.path.basename(f).split(".")[0]
        df = pd.read_csv(f)
        df["lang"] = lang
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    # Create a combined stratify key (polarization + language) to preserve both distributions
    stratify_col = df["polarization"].astype(str) + "_" + df["lang"].astype(str)
    return train_test_split(
        df,
        test_size=0.15,
        stratify=stratify_col,
        random_state=SEED
    )

def load_dev_data():
    data = {}
    for f in glob.glob(f"{DEV_DIR}/*.csv"):
        lang = os.path.basename(f).split(".")[0]
        df = pd.read_csv(f)
        df["lang"] = lang
        data[lang] = df
    return data

class PolarDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = df["polarization"].tolist()
        self.langs = df["lang"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.langs[idx]

# ---------- MODEL ----------
class XLMRContrastive(nn.Module):
    def __init__(self, model_name, num_clusters):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        self.cluster_emb = nn.Embedding(num_clusters, 32)

        self.projection = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.classifier = nn.Linear(
            self.encoder.config.hidden_size + 32, 2
        )

    def mean_pool(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        return (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, cluster_ids):
        out = self.encoder(input_ids, attention_mask)
        sent_emb = self.mean_pool(out.last_hidden_state, attention_mask)

        proj = self.projection(sent_emb)
        cemb = self.cluster_emb(cluster_ids)

        logits = self.classifier(torch.cat([sent_emb, cemb], dim=-1))
        return logits, proj

# ---------- SUPERVISED CONTRASTIVE ----------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, feats, labels):
        feats = nn.functional.normalize(feats, dim=1)
        sim = torch.matmul(feats, feats.T) / self.temperature

        labels = labels.unsqueeze(1)
        mask = (labels == labels.T).float().to(feats.device)
        logits_mask = torch.eye(len(labels)).to(feats.device)
        mask = mask * (1 - logits_mask)

        exp_sim = torch.exp(sim) * (1 - logits_mask)
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-9)

        mean_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        return -mean_log_prob.mean()

# ---------- COLLATE ----------
def collate(batch, tokenizer):
    texts, labels, langs = zip(*batch)
    cluster_ids = torch.tensor(
        [CLUSTER2ID[LANG2CLUSTER[l]] for l in langs]
    )

    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"], torch.tensor(labels), cluster_ids

# ---------- TRAIN + EVAL ----------
def train_and_validate():
    train_df, val_df = load_train_data()
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL, use_fast=True)

    train_loader = DataLoader(
        PolarDataset(train_df),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: collate(x, tokenizer)
    )

    val_loader = DataLoader(
        PolarDataset(val_df),
        batch_size=8,
        shuffle=False,
        collate_fn=lambda x: collate(x, tokenizer)
    )

    model = XLMRContrastive(ENCODER_MODEL, len(CLUSTER2ID)).to(DEVICE)
    con_loss = SupConLoss(TEMPERATURE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(0.1 * EPOCHS * len(train_loader)),
        EPOCHS * len(train_loader)
    )

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        total_loss = 0

        for step, (ids, mask, labels, cids) in enumerate(tqdm(train_loader)):
            ids, mask = ids.to(DEVICE), mask.to(DEVICE)
            labels, cids = labels.to(DEVICE), cids.to(DEVICE)

            logits, proj = model(ids, mask, cids)
            ce = nn.functional.cross_entropy(logits, labels)
            cl = con_loss(proj, labels)

            loss = (ce + 0.3 * cl) / ACCUM_STEPS
            loss.backward()

            if (step + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        # ---------- VALIDATION ----------
        model.eval()
        y_true, y_pred, probs = [], [], []

        with torch.no_grad():
            for ids, mask, labels, cids in val_loader:
                ids, mask = ids.to(DEVICE), mask.to(DEVICE)
                cids = cids.to(DEVICE)

                logits, _ = model(ids, mask, cids)
                prob = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                preds = torch.argmax(logits, dim=1).cpu().tolist()

                probs.extend(prob)
                y_pred.extend(preds)
                y_true.extend(labels.tolist())

        macro_f1 = f1_score(y_true, y_pred, average="macro")
        print(f"\n🔥 Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Macro-F1: {macro_f1:.4f}\n")

    val_df = val_df.reset_index(drop=True)
    val_df["prob"] = probs
    return model, tokenizer, val_df

# ---------- THRESHOLD TUNING ----------
def tune_thresholds(val_df):
    thresholds = {}
    best_global = 0.0

    for lang in val_df.lang.unique():
        best_f1, best_t = 0, 0.5
        subset = val_df[val_df.lang == lang]

        for t in np.arange(0.2, 0.8, 0.02):
            preds = (subset.prob > t).astype(int)
            f1 = f1_score(subset.polarization, preds, average="binary")
            if f1 > best_f1:
                best_f1, best_t = f1, t

        thresholds[lang] = best_t

    return thresholds

# ---------- PREDICTION ----------
@torch.no_grad()
def predict_and_save(model, tokenizer, thresholds):
    model.eval()
    dev_sets = load_dev_data()

    for lang, df in dev_sets.items():
        preds = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Predict {lang}"):
            enc = tokenizer(
                row["text"],
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            ).to(DEVICE)

            cid = torch.tensor(
                [CLUSTER2ID[LANG2CLUSTER[lang]]]
            ).to(DEVICE)

            logits, _ = model(enc["input_ids"], enc["attention_mask"], cid)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            preds.append(1 if prob > thresholds.get(lang, 0.5) else 0)

        out = pd.DataFrame({
            "id": df["id"],
            "polarization": preds
        })
        out.to_csv(f"{PRED_DIR}/pred_{lang}.csv", index=False)

# ---------- RUN PIPELINE ----------
model, tokenizer, val_df = train_and_validate()
thresholds = tune_thresholds(val_df)
predict_and_save(model, tokenizer, thresholds)