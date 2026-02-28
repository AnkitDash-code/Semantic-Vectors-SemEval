# ==================================================================================
# SemEval 2026 Subtask-1 FINAL SYSTEM (PRODUCTION-STABLE)
# mDeBERTa + XLM-R | QLoRA | Attention Pooling | Focal Loss
# CV | Weighted Ensemble | Per-Language F1 | Threshold Tuning
# ==================================================================================

import os, gc, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    AutoModel,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)

from torch.optim import AdamW
from tqdm import tqdm

# ===================== ENV =====================
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ===================== CONFIG =====================
MODEL_LIST = [
    "microsoft/mdeberta-v3-base",
    "xlm-roberta-base"
]

EPOCHS     = 3        # 🔁 6–8 for final run
N_FOLDS    = 2        # 🔁 4–5 for final run
BATCH_SIZE = 32

LR      = 2e-4
MAX_LEN = 128
SEED    = 42

RANK  = 64
ALPHA = 128

# XLM-R is stronger for these
XLMR_BOOST_LANGS = {"hau", "hin", "pan", "ori"}

OUT_DIR = "/kaggle/working/predictions"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===================== UTILS =====================
def safe_model_name(name):
    return name.replace("/", "_")

# ===================== DATA =====================
def auto_find_files():
    file_map = {"train": [], "dev": []}
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f.endswith(".csv"):
                p = os.path.join(root, f).lower()
                if "train" in p:
                    file_map["train"].append(os.path.join(root, f))
                elif "dev" in p or "val" in p:
                    file_map["dev"].append(os.path.join(root, f))
    return file_map

FILE_MAP = auto_find_files()

dfs = []
for f in FILE_MAP["train"] + FILE_MAP["dev"]:
    lang = os.path.basename(f).split("_")[0]
    df = pd.read_csv(f)
    df["lang"] = lang
    dfs.append(df)

FULL_DF = pd.concat(dfs).reset_index(drop=True)

# ===================== DATASET =====================
class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df.text.tolist()
        self.labels = df.polarization.tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt"
        )
        return (
            enc.input_ids.squeeze(0),
            enc.attention_mask.squeeze(0),
            torch.tensor(self.labels[idx])
        )

# ===================== LOSS =====================
def focal_loss(logits, labels, gamma=2.0):
    ce = nn.CrossEntropyLoss(reduction="none")(logits, labels)
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()

# ===================== MODEL =====================
class QLoraClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.backbone = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map="auto"
        )

        self.backbone = prepare_model_for_kbit_training(self.backbone)

        peft_cfg = LoraConfig(
            r=RANK,
            lora_alpha=ALPHA,
            lora_dropout=0.1,
            bias="none",
            target_modules="all-linear",
            task_type=TaskType.FEATURE_EXTRACTION
        )

        self.backbone = get_peft_model(self.backbone, peft_cfg)

        h = self.backbone.config.hidden_size
        self.attn = nn.Linear(h, 1)
        self.drop = nn.Dropout(0.2)
        self.cls  = nn.Linear(h, 2)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        x = out.last_hidden_state
        a = self.attn(x).squeeze(-1)
        a = a.masked_fill(attention_mask == 0, -1e9)
        w = torch.softmax(a, dim=1)

        pooled = torch.sum(x * w.unsqueeze(-1), dim=1)
        logits = self.cls(self.drop(pooled))

        loss = focal_loss(logits, labels) if labels is not None else None
        return logits, loss

# ===================== TRAINING =====================
def train_model(model_name):
    print(f"\n🚀 Training {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    oof = FULL_DF.copy()
    oof["prob"] = 0.0

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    strat = FULL_DF.polarization.astype(str) + "_" + FULL_DF.lang

    for fold, (tr, va) in enumerate(skf.split(FULL_DF, strat)):
        print(f"\n🔁 Fold {fold+1}")

        train_dl = DataLoader(
            TextDataset(FULL_DF.iloc[tr], tokenizer),
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        val_dl = DataLoader(
            TextDataset(FULL_DF.iloc[va], tokenizer),
            batch_size=BATCH_SIZE * 2,
            shuffle=False
        )

        model = QLoraClassifier(model_name).to(DEVICE)
        opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

        sched = get_cosine_schedule_with_warmup(
            opt, 100, EPOCHS * len(train_dl)
        )

        best_f1 = 0.0
        best_state = None

        for ep in range(EPOCHS):
            model.train()
            for ids, mask, y in tqdm(train_dl, desc=f"Epoch {ep+1}"):
                ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
                _, loss = model(ids, mask, y)
                loss.backward()
                opt.step()
                opt.zero_grad()
                sched.step()

            model.eval()
            probs, refs = [], []
            with torch.no_grad():
                for ids, mask, y in val_dl:
                    ids, mask = ids.to(DEVICE), mask.to(DEVICE)
                    logits, _ = model(ids, mask)
                    probs += torch.softmax(logits, 1)[:,1].cpu().tolist()
                    refs += y.tolist()

            f1 = f1_score(refs, np.array(probs) > 0.5, average="macro")
            print(f"   F1 = {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        # ✅ restore BEST epoch (QLoRA-safe)
        model.load_state_dict(best_state, strict=False)
        print(f"   ✔ Using BEST F1 = {best_f1:.4f}")

        model.eval()
        with torch.no_grad():
            probs = []
            for ids, mask, _ in val_dl:
                ids, mask = ids.to(DEVICE), mask.to(DEVICE)
                logits, _ = model(ids, mask)
                probs += torch.softmax(logits, 1)[:,1].cpu().tolist()

        oof.loc[va, "prob"] = probs

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return oof

# ===================== TRAIN MODELS =====================
all_oof = []

for m in MODEL_LIST:
    oof = train_model(m)
    oof.rename(columns={"prob": f"prob_{safe_model_name(m)}"}, inplace=True)
    all_oof.append(oof)

# ===================== MERGE =====================
final_oof = all_oof[0]
for df in all_oof[1:]:
    final_oof = final_oof.join(df[[c for c in df.columns if c.startswith("prob_")]])

# ===================== WEIGHTED ENSEMBLE =====================
pcols = [c for c in final_oof.columns if c.startswith("prob_")]
mcol = [c for c in pcols if "mdeberta" in c][0]
xcol = [c for c in pcols if "xlm" in c][0]

def ensemble(r):
    return (
        0.4 * r[mcol] + 0.6 * r[xcol]
        if r.lang in XLMR_BOOST_LANGS
        else 0.6 * r[mcol] + 0.4 * r[xcol]
    )

final_oof["prob"] = final_oof.apply(ensemble, axis=1)

# ===================== PER-LANG F1 =====================
print("\n📊 Per-language F1")
for lang in sorted(final_oof.lang.unique()):
    sub = final_oof[final_oof.lang == lang]
    print(f"{lang:>5} : {f1_score(sub.polarization, sub.prob > 0.5, average='macro'):.4f}")

# ===================== THRESHOLD TUNING =====================
print("\n⚖️ Threshold tuning")
final_oof["pred"] = 0

for lang in final_oof.lang.unique():
    sub = final_oof[final_oof.lang == lang]
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.2, 0.8, 0.02):
        f1 = f1_score(sub.polarization, sub.prob > t, average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    final_oof.loc[sub.index, "pred"] = (sub.prob > best_t).astype(int)
    print(f"{lang:>5} | t={best_t:.2f} | f1={best_f1:.4f}")

# ===================== FINAL SCORE =====================
print(
    "\n🏆 FINAL CV MACRO F1:",
    f1_score(final_oof.polarization, final_oof.pred, average="macro")
)