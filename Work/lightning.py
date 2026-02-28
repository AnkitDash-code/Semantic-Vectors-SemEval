# ==========================================================
# SemEval 2026 FINAL SYSTEM (Lightning AI Version)
# - mDeBERTa + XLM-R base models
# - DE/IT/SW expert models (masked by language)
# - QLoRA (4-bit) + LoRA adapters
# - Stacker with proper Cross-Validation (No Leakage)
# ==========================================================

import os
import gc
import random
import warnings
from copy import deepcopy
from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

# Transformers, BitsAndBytes & PEFT
from transformers import (
    AutoTokenizer,
    AutoModel,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# SWA
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim import AdamW

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==================== DEBUG CONTROL ====================
# Set to True for a quick 1-minute test run to verify code works.
# Set to False for the actual training session.
DEBUG_MODE = False

if DEBUG_MODE:
    print("\n⚠️ WARNING: DEBUG MODE IS ON - RUNNING ON TINY DATASET ⚠️")

# ==================== CONFIG ====================
MODEL_BASES = [
    "microsoft/mdeberta-v3-base",
    "xlm-roberta-base"
]

# Expert model IDs
EXPERT_MODELS = {
    "deu": "dbmdz/bert-base-german-cased",       # German expert
    "ita": "dbmdz/bert-base-italian-cased",      # Italian expert
    "swa": "Davlan/bert-base-multilingual-cased-finetuned-swahili"  # Swahili
}

# HYPERPARAMETERS
EPOCHS_BASE = 1 if DEBUG_MODE else 4      
EPOCHS_EXPERT = 1 if DEBUG_MODE else 4        
FOLDS = 2 if DEBUG_MODE else 3
BATCH = 4 if DEBUG_MODE else 32
MAX_LEN = 128
LR_MAP = {
    "microsoft/mdeberta-v3-base": 2e-4,
    "xlm-roberta-base": 5e-5,
}
RANDOM_SEED = 42

# SWA SETTINGS - DISABLED for 4-bit stability
USE_SWA = False   # Set False to prevent 4-bit averaging crashes
SWA_START = 2
SWA_LR = None

# --- PATHS UPDATED FOR LIGHTNING AI ---
OUT_DIR = "./subtask1"
os.makedirs(OUT_DIR, exist_ok=True)

# Hardware check for bfloat16
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

BNB_CONFIG = dict(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype
)

LORA_R = 32
LORA_ALPHA = 64

FOCAL_GAMMA = 2.0
LABEL_SMOOTH = 0.05

# Stacker settings
STACKER_EARLY_STOP = True
STACKER_ES_ROUNDS = 50
STACKER_NUM_ESTIMATORS = 10 if DEBUG_MODE else 1000
STACKER_LR = 0.05
STACKER_MAX_DEPTH = 4

# Reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE, "| Compute Dtype:", compute_dtype)

# ==================== UTILITIES ====================
def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")

def cpu_state_dict_clone(model: torch.nn.Module) -> Dict:
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def try_makedirs(path):
    os.makedirs(path, exist_ok=True)

# ==================== DATA LOADING ====================
# Updated for Lightning AI local folder structure
def auto_find_files(root="./data"):
    file_map = {"train": [], "dev": [], "test": []}
    if not os.path.exists(root):
        print(f"ERROR: Data directory '{root}' not found. Please create it and upload CSVs.")
        return file_map

    for rootdir, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            p = os.path.join(rootdir, f)
            pl = p.lower()
            if "train" in pl:
                file_map["train"].append(p)
            elif "dev" in pl or "val" in pl:
                file_map["dev"].append(p)
            elif "test" in pl:
                file_map["test"].append(p)
    return file_map

FILE_MAP = auto_find_files()
if len(FILE_MAP["train"]) + len(FILE_MAP["dev"]) == 0:
    print("WARNING: No train/dev CSVs found. Ensure dataset is in './data' folder.")
    # Create dummy DF for syntax checking if running without data
    TRAIN_DF = pd.DataFrame(columns=["id", "text", "lang", "polarization"])
    TEST_DF = pd.DataFrame(columns=["id", "text", "lang"])
else:
    def load_concat(files):
        dfs = []
        for f in files:
            df = pd.read_csv(f)
            # Robust language parsing: try 'lang' col, else filename
            if "lang" not in df.columns:
                # heuristic: filename like 'train_de.csv' -> 'de'
                # or 'de_train.csv' -> 'de'
                base = os.path.splitext(os.path.basename(f))[0]
                parts = base.split("_")
                # assume lang code is the length-3 part if available
                found = False
                for p in parts:
                    if len(p) == 3 and p.isalpha(): 
                        df["lang"] = p
                        found = True
                        break
                if not found:
                    df["lang"] = parts[0] # Fallback
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    TRAIN_DF = load_concat(FILE_MAP["train"] + FILE_MAP["dev"])
    TEST_DF = load_concat(FILE_MAP["test"]) if FILE_MAP["test"] else pd.DataFrame(columns=TRAIN_DF.columns)

    # --- DEBUG SLICING ---
    if DEBUG_MODE:
        print("Debug: Truncating datasets for speed test...")
        TRAIN_DF = TRAIN_DF.head(64)
        if len(TEST_DF) > 0:
            TEST_DF = TEST_DF.head(20)

print("Loaded Train:", len(TRAIN_DF), "Test:", len(TEST_DF))
if len(TRAIN_DF) > 0:
    print("Languages:", TRAIN_DF.lang.unique().tolist())

# ==================== DATASET ====================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        item = {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0)
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ==================== LOSS ====================
class FocalLossWithSmoothing(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        B, C = logits.size()
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.label_smoothing / (C - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        log_preds = F.log_softmax(logits, dim=1)
        ce = -(true_dist * log_preds).sum(dim=1)
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

focal_loss_fn = FocalLossWithSmoothing(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTH)

# ==================== MODEL CLASS ====================
class QLoraClassifier(nn.Module):
    def __init__(self, model_name: str, lora_r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=None):
        super().__init__()
        bnb_conf = BitsAndBytesConfig(**BNB_CONFIG)
        self.backbone = AutoModel.from_pretrained(
            model_name,
            quantization_config=bnb_conf,
            device_map="auto"
        )
        self.backbone = prepare_model_for_kbit_training(self.backbone)

        # --- FIX: Correct target modules to avoid crashes ---
        if target_modules is None:
            name_low = model_name.lower()
            if "deberta" in name_low or "deberta-v3" in name_low:
                # Removed 'output' to fix ValueError
                target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
            elif "roberta" in name_low or "xlm-roberta" in name_low:
                target_modules = ["query", "key", "value", "dense"]
            else:
                target_modules = ["query", "key", "value", "dense"]

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        self.backbone = get_peft_model(self.backbone, peft_config)

        hidden = self.backbone.config.hidden_size
        self.attn = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state
        
        # Attention pooling
        attn_scores = self.attn(last_hidden).squeeze(-1)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.sum(last_hidden * attn_weights.unsqueeze(-1), dim=1)
        
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss = focal_loss_fn(logits, labels)
        return logits, loss

# ==================== TRAIN FUNCTION (OOF) ====================
def train_model_oof(model_name: str, df: pd.DataFrame, folds=FOLDS, epochs=EPOCHS_BASE, batch=BATCH, lr=None):
    print(f"\n>>> Train OOF for {model_name}")
    safe = safe_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    result_df = df.copy()
    result_df["prob"] = 0.0

    strat_col = df.polarization.astype(str) + "_" + df.lang.astype(str)
    
    # Handle tiny datasets in debug mode where stratify might fail
    try:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
        splits = list(skf.split(df, strat_col))
    except ValueError:
        print("Warning: StratifiedKFold failed (probably mostly 1 class in Debug mode). Using standard KFold.")
        from sklearn.model_selection import KFold
        skf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
        splits = list(skf.split(df))

    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\n--- Fold {fold+1}/{folds} for {model_name} ---")
        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        va_df = df.iloc[va_idx].reset_index(drop=True)

        train_ds = TextDataset(tr_df.text.tolist(), tr_df.polarization.tolist(), tokenizer)
        val_ds = TextDataset(va_df.text.tolist(), va_df.polarization.tolist(), tokenizer)

        train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=batch*2, shuffle=False, num_workers=2, pin_memory=True)

        model = QLoraClassifier(model_name).to(DEVICE)
        opt = AdamW(model.parameters(), lr=(lr if lr is not None else LR_MAP.get(model_name, 2e-4)), weight_decay=0.01)
        
        total_steps = epochs * len(train_dl)
        warmup_steps = max(1, int(0.1 * total_steps))
        sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
        
        # SWA setup (optional)
        swa_model = AveragedModel(model) if USE_SWA else None
        swa_scheduler = SWALR(opt, swa_lr=opt.param_groups[0]['lr'] / 2) if USE_SWA else None

        best_f1 = -1.0
        best_state = None

        for ep in range(epochs):
            model.train()
            loop = tqdm(train_dl, desc=f"Epoch {ep+1}/{epochs}", leave=False)
            for batch_data in loop:
                ids = batch_data["input_ids"].to(DEVICE)
                mask = batch_data["attention_mask"].to(DEVICE)
                labels = batch_data["labels"].to(DEVICE)

                logits, loss = model(ids, mask, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                if not USE_SWA or ep < SWA_START:
                    sched.step()

            if USE_SWA and ep >= SWA_START:
                swa_model.update_parameters(model)
                swa_scheduler.step()

            # Validation
            model.eval()
            probs, refs = [], []
            with torch.no_grad():
                for batch_data in val_dl:
                    ids = batch_data["input_ids"].to(DEVICE)
                    mask = batch_data["attention_mask"].to(DEVICE)
                    logits, _ = model(ids, mask)
                    p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
                    probs += p
                    refs += batch_data["labels"].to(DEVICE).cpu().tolist()

            f1 = f1_score(refs, (np.array(probs) > 0.5).astype(int), average="macro")
            print(f" Fold {fold+1} Ep {ep+1} F1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_state = cpu_state_dict_clone(model)

        # Load best weights
        if best_state is not None:
            model.load_state_dict(best_state, strict=False)

        # OOF Prediction
        model.eval()
        val_probs = []
        with torch.no_grad():
            for batch_data in val_dl:
                ids = batch_data["input_ids"].to(DEVICE)
                mask = batch_data["attention_mask"].to(DEVICE)
                logits, _ = model(ids, mask)
                p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
                val_probs += p
        result_df.loc[va_idx, "prob"] = val_probs

        # Save Checkpoint
        fname = f"{OUT_DIR}/{safe}_fold{fold}.pt"
        torch.save(best_state if best_state else cpu_state_dict_clone(model), fname)
        
        del model
        torch.cuda.empty_cache()
        gc.collect()

    return result_df, tokenizer

# ==================== EXPERT TRAINING (Masked) ====================
def train_expert_oof(lang_code: str, model_id: str, df: pd.DataFrame, train_expert: bool=True):
    print(f"\n>>> Expert {lang_code}: {model_id} (train={train_expert})")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    sub_df = df[df.lang == lang_code].reset_index(drop=True)
    
    if len(sub_df) == 0:
        tmp = df.copy()
        tmp[f"prob_expert_{lang_code}"] = 0.0
        return tmp[["id", f"prob_expert_{lang_code}"]], tokenizer

    if not train_expert:
        # Inference only (Pretrained)
        ds = TextDataset(sub_df.text.tolist(), sub_df.polarization.tolist(), tokenizer)
        dl = DataLoader(ds, batch_size=BATCH*2, shuffle=False, num_workers=2)
        model = QLoraClassifier(model_id).to(DEVICE)
        model.eval()
        probs = []
        with torch.no_grad():
            for batch_data in dl:
                ids = batch_data["input_ids"].to(DEVICE)
                mask = batch_data["attention_mask"].to(DEVICE)
                logits, _ = model(ids, mask)
                probs += torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
        
        out = sub_df.copy()
        out[f"prob_expert_{lang_code}"] = probs
        full = df.copy().merge(out[["id", f"prob_expert_{lang_code}"]], on="id", how="left")
        # FILL MISSING WITH -1 to indicate "Expert did not run here"
        full[f"prob_expert_{lang_code}"] = full[f"prob_expert_{lang_code}"].fillna(-1.0)
        return full[["id", f"prob_expert_{lang_code}"]], tokenizer

    # Training Expert (K-Fold on language subset)
    result = sub_df.copy()
    result[f"prob_expert_{lang_code}"] = 0.0
    
    n_splits = min(FOLDS, len(sub_df))
    if n_splits < 2:
        # Too few samples to fold, just predict 0.5 (or skip)
        print(f"Warning: Not enough data for {lang_code} folding. Returning defaults.")
        full = df.copy()
        full[f"prob_expert_{lang_code}"] = -1.0
        return full[["id", f"prob_expert_{lang_code}"]], tokenizer

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    # Stratified split catch for tiny data
    try:
        splits = list(skf.split(sub_df, sub_df.polarization))
    except ValueError:
        from sklearn.model_selection import KFold
        skf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        splits = list(skf.split(sub_df))

    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f" Expert {lang_code} fold {fold+1}/{n_splits}")
        tr_df = sub_df.iloc[tr_idx].reset_index(drop=True)
        va_df = sub_df.iloc[va_idx].reset_index(drop=True)
        
        train_ds = TextDataset(tr_df.text.tolist(), tr_df.polarization.tolist(), tokenizer)
        val_ds = TextDataset(va_df.text.tolist(), va_df.polarization.tolist(), tokenizer)
        train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
        val_dl = DataLoader(val_ds, batch_size=BATCH*2, shuffle=False, num_workers=2)
        
        model = QLoraClassifier(model_id).to(DEVICE)
        opt = AdamW(model.parameters(), lr=LR_MAP.get(model_id, 2e-5), weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, int(0.1*EPOCHS_EXPERT*len(train_dl)), EPOCHS_EXPERT*len(train_dl))
        
        best_f1 = -1.0
        best_state = None
        
        for ep in range(EPOCHS_EXPERT):
            model.train()
            for batch_data in tqdm(train_dl, desc=f"Ep{ep+1}", leave=False):
                ids = batch_data["input_ids"].to(DEVICE)
                mask = batch_data["attention_mask"].to(DEVICE)
                labels = batch_data["labels"].to(DEVICE)
                logits, loss = model(ids, mask, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                sched.step()
                
            model.eval()
            probs, refs = [], []
            with torch.no_grad():
                for batch_data in val_dl:
                    ids = batch_data["input_ids"].to(DEVICE)
                    mask = batch_data["attention_mask"].to(DEVICE)
                    logits, _ = model(ids, mask)
                    probs += torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
                    refs += batch_data["labels"].cpu().tolist()
            f1 = f1_score(refs, (np.array(probs) > 0.5).astype(int), average="macro")
            
            if f1 > best_f1:
                best_f1 = f1
                best_state = cpu_state_dict_clone(model)
                
        if best_state is not None:
            model.load_state_dict(best_state, strict=False)
            
        # OOF
        model.eval()
        f_probs = []
        with torch.no_grad():
            for batch_data in val_dl:
                ids = batch_data["input_ids"].to(DEVICE)
                mask = batch_data["attention_mask"].to(DEVICE)
                logits, _ = model(ids, mask)
                f_probs += torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
        result.loc[va_idx, f"prob_expert_{lang_code}"] = f_probs
        
        # Save
        torch.save(best_state, f"{OUT_DIR}/expert_{lang_code}_fold{fold}.pt")
        del model; torch.cuda.empty_cache(); gc.collect()
        
    full = df.copy().merge(result[["id", f"prob_expert_{lang_code}"]], on="id", how="left")
    full[f"prob_expert_{lang_code}"] = full[f"prob_expert_{lang_code}"].fillna(-1.0)
    return full[["id", f"prob_expert_{lang_code}"]], tokenizer

# ==================== RUN MODELS (OOF) ====================
if len(TRAIN_DF) > 0:
    base_oofs = []
    base_tokenizers = {}
    for base_model in MODEL_BASES:
        df_oof, tok = train_model_oof(base_model, TRAIN_DF)
        col_name = f"prob_{safe_name(base_model)}"
        df_oof = df_oof[["id", "lang", "text", "polarization", "prob"]].rename(columns={"prob": col_name})
        base_oofs.append(df_oof)
        base_tokenizers[base_model] = tok

    # Merge Base OOFs
    meta_df = base_oofs[0].set_index("id")
    for df_o in base_oofs[1:]:
        meta_df = meta_df.join(df_o.set_index("id")[[c for c in df_o.columns if c.startswith("prob_")]])
    meta_df = meta_df.reset_index()

    # Run Experts
    expert_oofs = []
    expert_tokenizers = {}
    for lang_code, model_id in EXPERT_MODELS.items():
        train_expert = (lang_code != "swa") 
        df_expert_oof, tok = train_expert_oof(lang_code, model_id, TRAIN_DF, train_expert=train_expert)
        expert_oofs.append(df_expert_oof[["id", f"prob_expert_{lang_code}"]])
        expert_tokenizers[lang_code] = tok

    # Join Experts
    for df_e in expert_oofs:
        meta_df = meta_df.merge(df_e, on="id", how="left")
    meta_df.fillna(-1.0, inplace=True) # Fill missing experts with -1

    # ==================== STACKER (FIXED LEAKAGE) ====================
    print("\n\n======= STACKER TRAINING & CV =======")
    prob_cols = [c for c in meta_df.columns if c.startswith("prob_")]
    expert_cols = [c for c in meta_df.columns if c.startswith("prob_expert_")]
    
    meta_df["text_len"] = meta_df["text"].str.len()
    meta_df["max_p"] = meta_df[prob_cols + expert_cols].max(axis=1)
    # Mask negative values (from -1 imputation) before log for entropy
    valid_probs = meta_df[prob_cols + expert_cols].clip(lower=0.0)
    meta_df["entropy"] = - (valid_probs * np.log(valid_probs + 1e-9)).sum(axis=1)

    FEAT_COLS = prob_cols + expert_cols + ["text_len", "max_p", "entropy"]
    X = meta_df[FEAT_COLS].values
    y = meta_df["polarization"].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Generate UNBIASED Stacker OOF for Calibration
    meta_df["stack_prob_oof"] = 0.0
    skf_stack = StratifiedKFold(n_splits=5 if not DEBUG_MODE else 2, shuffle=True, random_state=RANDOM_SEED)
    
    xgb_params = {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "eta": STACKER_LR, "max_depth": STACKER_MAX_DEPTH,
        "subsample": 0.9, "colsample_bytree": 0.9,
        "verbosity": 0, "seed": RANDOM_SEED
    }

    print("Generating Stacker OOF for Calibration...")
    try:
        split_gen = skf_stack.split(X_scaled, y)
    except ValueError:
         # Debug mode fallback
        skf_stack = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
        split_gen = skf_stack.split(X_scaled, y)

    for tr_ix, val_ix in split_gen:
        X_tr, X_val = X_scaled[tr_ix], X_scaled[val_ix]
        y_tr = y[tr_ix]
        
        d_tr = xgb.DMatrix(X_tr, label=y_tr)
        d_val = xgb.DMatrix(X_val)
        
        bst = xgb.train(xgb_params, d_tr, num_boost_round=STACKER_NUM_ESTIMATORS)
        preds = bst.predict(d_val)
        meta_df.iloc[val_ix, meta_df.columns.get_loc("stack_prob_oof")] = preds

    # 2. Fit Final Stacker on ALL Data (for Test Inference)
    print("Training Final Stacker on All Data...")
    d_full = xgb.DMatrix(X_scaled, label=y)
    final_stacker = xgb.train(xgb_params, d_full, num_boost_round=STACKER_NUM_ESTIMATORS)

    # 3. Fit Calibrators & Thresholds on UNBIASED OOF
    print("Calibrating on OOF probs...")
    language_thresholds = {}
    language_calibrators = {}

    for lang in sorted(meta_df.lang.unique()):
        sub = meta_df[meta_df.lang == lang]
        if len(sub) < 10:
            language_thresholds[lang] = 0.5
            language_calibrators[lang] = None
            continue
            
        lr = LogisticRegression(solver="liblinear")
        X_lr = sub[["stack_prob_oof"]].values.reshape(-1, 1)
        lr.fit(X_lr, sub["polarization"].values)
        language_calibrators[lang] = lr
        
        # Find best threshold on OOF
        probs_cal = lr.predict_proba(X_lr)[:, 1]
        best_t, best_f1 = 0.5, -1
        for t in np.linspace(0.1, 0.9, 81):
            preds = (probs_cal > t).astype(int)
            f = f1_score(sub["polarization"].values, preds, average="macro")
            if f > best_f1:
                best_f1 = f
                best_t = t
        language_thresholds[lang] = best_t
        print(f" Lang {lang}: Thr={best_t:.3f}, OOF F1={best_f1:.4f}")

    # Save Thresholds
    pd.DataFrame.from_dict(language_thresholds, orient="index", columns=["threshold"]).to_csv(f"{OUT_DIR}/language_thresholds.csv")

# ==================== FINAL FULL TRAINING (BASE MODELS) ====================
print("\n======= FULL TRAINING BASE MODELS =======")
final_models = {}

def train_full(model_name, df, epochs, batch):
    print(f"Full Train: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    ds = TextDataset(df.text.tolist(), df.polarization.tolist(), tok)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2)
    
    model = QLoraClassifier(model_name).to(DEVICE)
    opt = AdamW(model.parameters(), lr=LR_MAP.get(model_name, 2e-4), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(0.1*epochs*len(dl)), epochs*len(dl))
    
    for ep in range(epochs):
        model.train()
        for batch_data in tqdm(dl, desc=f"Ep{ep+1}", leave=False):
            ids = batch_data["input_ids"].to(DEVICE)
            mask = batch_data["attention_mask"].to(DEVICE)
            labels = batch_data["labels"].to(DEVICE)
            logits, loss = model(ids, mask, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            sched.step()
    return model, tok

if len(TRAIN_DF) > 0:
    for base in MODEL_BASES:
        m, t = train_full(base, TRAIN_DF, EPOCHS_BASE, BATCH)
        final_models[base] = m
        base_tokenizers[base] = t
        torch.save(cpu_state_dict_clone(m), f"{OUT_DIR}/{safe_name(base)}_full.pt")

    # Experts Full
    for lang_code, model_id in EXPERT_MODELS.items():
        if lang_code == "swa":
            # Just load pretrained
            m = QLoraClassifier(model_id).to(DEVICE)
            t = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            final_models[f"expert_{lang_code}"] = m
            expert_tokenizers[lang_code] = t
        else:
            sub = TRAIN_DF[TRAIN_DF.lang == lang_code].reset_index(drop=True)
            if len(sub) > 0:
                m, t = train_full(model_id, sub, EPOCHS_EXPERT, BATCH)
                final_models[f"expert_{lang_code}"] = m
                expert_tokenizers[lang_code] = t
                torch.save(cpu_state_dict_clone(m), f"{OUT_DIR}/expert_{lang_code}_full.pt")

# ==================== INFERENCE ON TEST SET ====================
if len(TEST_DF) > 0 and len(TRAIN_DF) > 0:
    print("\n======= TEST INFERENCE =======")
    test_res = TEST_DF.copy()
    
    def get_preds(model, tok, texts):
        ds = TextDataset(texts, [0]*len(texts), tok)
        dl = DataLoader(ds, batch_size=BATCH*2, shuffle=False)
        model.eval()
        probs = []
        with torch.no_grad():
            for batch_data in dl:
                ids = batch_data["input_ids"].to(DEVICE)
                mask = batch_data["attention_mask"].to(DEVICE)
                logits, _ = model(ids, mask)
                probs += torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
        return np.array(probs)

    # Base Inference
    for base in MODEL_BASES:
        print(f"Infer: {base}")
        test_res[f"prob_{safe_name(base)}"] = get_preds(final_models[base], base_tokenizers[base], test_res.text.tolist())

    # Expert Inference (Masked)
    for lang_code, model_id in EXPERT_MODELS.items():
        col = f"prob_expert_{lang_code}"
        test_res[col] = -1.0
        mask = test_res.lang == lang_code
        if mask.sum() > 0:
            print(f"Infer Expert: {lang_code}")
            # Use trained expert if available, else skip
            if f"expert_{lang_code}" in final_models:
                m = final_models[f"expert_{lang_code}"]
                t = expert_tokenizers[lang_code]
                subset_texts = test_res.loc[mask, "text"].tolist()
                preds = get_preds(m, t, subset_texts)
                test_res.loc[mask, col] = preds

    # Meta Features
    test_res["text_len"] = test_res.text.str.len()
    prob_cols = [c for c in test_res.columns if c.startswith("prob_")]
    valid_probs = test_res[prob_cols].clip(lower=0.0) # treat -1 as 0 for max/entropy
    test_res["max_p"] = valid_probs.max(axis=1)
    test_res["entropy"] = - (valid_probs * np.log(valid_probs + 1e-9)).sum(axis=1)

    # Scale & Predict
    X_test = scaler.transform(test_res[FEAT_COLS].values)
    test_res["stack_prob"] = final_stacker.predict(xgb.DMatrix(X_test))

    # Apply Calibration & Thresholds
    test_res["prediction"] = 0
    for lang in test_res.lang.unique():
        idxs = test_res.lang == lang
        # Calibrate
        calibrator = language_calibrators.get(lang)
        thr = language_thresholds.get(lang, 0.5)
        
        if calibrator:
            raw = test_res.loc[idxs, ["stack_prob"]].values.reshape(-1, 1)
            cal_prob = calibrator.predict_proba(raw)[:, 1]
        else:
            cal_prob = test_res.loc[idxs, "stack_prob"].values
            
        test_res.loc[idxs, "prediction"] = (cal_prob > thr).astype(int)

    # Save
    for lang in test_res.lang.unique():
        sub = test_res[test_res.lang == lang][["id", "prediction"]]
        sub.to_csv(f"{OUT_DIR}/pred_{lang}.csv", index=False)
        print(f"Saved {OUT_DIR}/pred_{lang}.csv")
    
    print("ALL DONE.")
else:
    print("Skipping inference (No Test Data found or Train failed).")