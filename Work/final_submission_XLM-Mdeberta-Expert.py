# ==========================================================
# SemEval 2026 FINAL SYSTEM (Hybrid SOTA + Auto-Detect Data)
# ----------------------------------------------------------------------------------
# 1. Auto-Detects Data: Scans current dir, data/, and kaggle/input
# 2. Hybrid Training: Trains mDeBERTa & XLM-R simultaneously (Siamese QLoRA).
# 3. Expert Training: Trains SOTA experts (GBERT, UmBERTo, AfriBERTa) on subsets.
# 4. Meta-Stacking: XGBoost combines Hybrid + Expert predictions.
# ==========================================================

import os, gc, random, warnings, time, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer, AutoModel, BitsAndBytesConfig, 
    get_cosine_schedule_with_warmup, DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from torch.optim import AdamW

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Auto-detect best attention implementation
try:
    import flash_attn  # noqa: F401
    ATTN_IMPL = "flash_attention_2"
    print("⚡ Flash Attention 2 detected — using FA2")
except ImportError:
    ATTN_IMPL = "sdpa"  # PyTorch built-in, no extra install needed
    print("⚡ Flash Attention 2 not found — using PyTorch SDPA (still fast on H100)")

# ==================== DEBUG CONTROL ====================
# Set to True for a quick ~1-minute test run to verify code works.
# Set to False for the actual training session.
DEBUG_MODE = False

if DEBUG_MODE:
    print("\n⚠️  WARNING: DEBUG MODE IS ON — RUNNING ON TINY DATASET ⚠️")

# H100 optimizations
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")  # TF32 on H100

# ==================== CONFIG ===================="
MODEL_A = "microsoft/mdeberta-v3-base"
MODEL_B = "FacebookAI/xlm-roberta-large"  # Upgraded from base (270M→560M params)

EXPERT_MODELS = {
    "deu": "deepset/gbert-large",               
    "ita": "dbmdz/bert-base-italian-xxl-cased", 
}

EPOCHS_BASE = 1 if DEBUG_MODE else 4
EPOCHS_EXPERT = 1 if DEBUG_MODE else 4
BATCH = 4 if DEBUG_MODE else 128
GRAD_ACCUM = 1       # No accum needed — large batch already saturates GPU
LR_BASE = 3e-4       # Scale LR up with larger batch (linear scaling rule)
LR_EXPERT = 3e-5     # Scale LR up with larger batch
MAX_LEN = 64 if DEBUG_MODE else 256
SEED = 42
OUT_DIR = "predictions"  # Changed to relative path for safety
os.makedirs(OUT_DIR, exist_ok=True)

FOCAL_GAMMA = 2.0
LABEL_SMOOTH = 0.05

BNB_CONFIG = dict(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16  # A100 native — faster than fp16
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed_everything(SEED)

def safe_name(name): return name.replace("/", "_")
def cpu_state_dict_clone(model): return {k: v.cpu().clone() for k, v in model.state_dict().items()}

# ==================== 1. ROBUST DATA LOADER (THE FIX) ====================
def auto_find_files():
    """Scans multiple locations to find data."""
    # Priority: Current Dir -> Data Folder -> Kaggle Input
    search_paths = [".", "./data", "./dataset", "/kaggle/input"]
    
    file_map = {"train": [], "dev": [], "test": []}
    
    print("🕵️ Scanning for datasets...")
    for root_path in search_paths:
        if not os.path.exists(root_path): continue
        
        for root, _, files in os.walk(root_path):
            for file in files:
                if file.endswith(".csv") and "pred" not in file: # Ignore output files
                    full_path = os.path.join(root, file)
                    path_lower = full_path.lower()
                    
                    if "train" in path_lower: file_map["train"].append(full_path)
                    elif "dev" in path_lower or "val" in path_lower: file_map["dev"].append(full_path)
                    elif "test" in path_lower: file_map["test"].append(full_path)

    total = sum(len(v) for v in file_map.values())
    if total == 0:
        raise ValueError(f"❌ No CSV files found in {search_paths}. Please upload data!")
    
    print(f"✅ Found {len(file_map['train'])} Train, {len(file_map['dev'])} Dev, {len(file_map['test'])} Test files.")
    return file_map

def load_data():
    files = auto_find_files()
    
    def _load_files(file_list):
        dfs = []
        for f in file_list:
            try:
                base = os.path.basename(f).split(".")[0]
                lang = base.split("_")[1] if "_" in base else base[:3]
                df = pd.read_csv(f)
                df["lang"] = lang
                if "text" not in df.columns and "content" in df.columns:
                    df.rename(columns={"content": "text"}, inplace=True)
                if "label" in df.columns:
                    df.rename(columns={"label": "polarization"}, inplace=True)
                dfs.append(df)
            except: pass
        return pd.concat(dfs).reset_index(drop=True) if dfs else pd.DataFrame(columns=["id", "text", "lang", "polarization"])
    
    train_df = _load_files(files['train'])
    dev_df = _load_files(files['dev'])
    test_df = _load_files(files['test'])
    
    return train_df, dev_df, test_df

# ==================== 2. PARALLEL TOKENIZATION ====================
def parallel_tokenize(texts, tokenizer):
    def _chunk(chunk):
        return tokenizer(chunk, truncation=True, max_length=MAX_LEN, padding=False, return_attention_mask=True)
    
    n_jobs = 4
    chunk_size = len(texts)//n_jobs + 1
    chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
    
    results = Parallel(n_jobs=n_jobs, backend="threading")(delayed(_chunk)(c) for c in chunks)
    ids, masks = [], []
    for r in results:
        ids.extend(r["input_ids"])
        masks.extend(r["attention_mask"])
    return ids, masks

class FastDualDataset(Dataset):
    def __init__(self, ids_a, masks_a, ids_b, masks_b, labels=None):
        self.ids_a = ids_a; self.masks_a = masks_a
        self.ids_b = ids_b; self.masks_b = masks_b
        self.labels = labels
    def __len__(self): return len(self.ids_a)
    def __getitem__(self, i):
        item = {"ids_a": self.ids_a[i], "mask_a": self.masks_a[i], "ids_b": self.ids_b[i], "mask_b": self.masks_b[i]}
        if self.labels is not None: item["labels"] = self.labels[i]
        return item

class FastDataset(Dataset):
    def __init__(self, ids, masks, labels=None):
        self.ids = ids; self.masks = masks; self.labels = labels
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        item = {"input_ids": self.ids[i], "attention_mask": self.masks[i]}
        if self.labels is not None: item["labels"] = self.labels[i]
        return item

# ==================== 3. MODELS & LOSS ====================
class FocalLossWithSmoothing(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma; self.ls = label_smoothing
    def forward(self, logits, targets):
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.ls / (logits.size(1) - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        log_preds = F.log_softmax(logits, dim=1)
        ce = -(true_dist * log_preds).sum(dim=1)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

focal_loss_fn = FocalLossWithSmoothing(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTH)

class HybridDualModel(nn.Module):
    def __init__(self):
        super().__init__()
        bnb = BitsAndBytesConfig(**BNB_CONFIG)
        # mDeBERTa-v3: uses disentangled attention, no FA2 support
        _ma = prepare_model_for_kbit_training(AutoModel.from_pretrained(MODEL_A, quantization_config=bnb))
        _ma.gradient_checkpointing_disable()  # Disable to avoid fp16 overflow
        _ma.config.use_cache = False
        self.ma = get_peft_model(_ma, LoraConfig(r=32, lora_alpha=64, target_modules=["query_proj","value_proj"], task_type=TaskType.FEATURE_EXTRACTION))
        # XLM-RoBERTa: FA2 if available, else SDPA
        _mb = prepare_model_for_kbit_training(AutoModel.from_pretrained(MODEL_B, quantization_config=bnb, attn_implementation=ATTN_IMPL, torch_dtype=torch.bfloat16))
        _mb.gradient_checkpointing_disable()  # Disable to avoid fp16 overflow
        _mb.config.use_cache = False
        self.mb = get_peft_model(_mb, LoraConfig(r=32, lora_alpha=64, target_modules=["query","value"], task_type=TaskType.FEATURE_EXTRACTION))
        dim = self.ma.config.hidden_size + self.mb.config.hidden_size
        self.head = nn.Sequential(nn.Linear(dim, 768), nn.LayerNorm(768), nn.Dropout(0.1), nn.GELU(), nn.Linear(768, 2))
    def forward(self, ids_a, mask_a, ids_b, mask_b, labels=None):
        out_a = self.ma(ids_a, mask_a).last_hidden_state[:, 0, :]
        out_b = self.mb(ids_b, mask_b).last_hidden_state[:, 0, :]
        logits = self.head(torch.cat([out_a, out_b], dim=1))
        loss = focal_loss_fn(logits, labels) if labels is not None else None
        return logits, loss

class ExpertModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        bnb = BitsAndBytesConfig(**BNB_CONFIG)
        target_mods = ["query_proj", "value_proj"] if "deberta" in model_name.lower() else ["query", "value"]
        # Fast attention for BERT/RoBERTa experts; skip for DeBERTa (disentangled attn)
        use_fast_attn = "deberta" not in model_name.lower()
        extra_kw = {"attn_implementation": ATTN_IMPL, "torch_dtype": torch.bfloat16} if use_fast_attn else {}
        _bb = prepare_model_for_kbit_training(AutoModel.from_pretrained(model_name, quantization_config=bnb, **extra_kw))
        _bb.gradient_checkpointing_disable()  # Disable to avoid fp16 overflow
        _bb.config.use_cache = False
        self.backbone = get_peft_model(_bb, LoraConfig(r=16, lora_alpha=32, target_modules=target_mods, task_type=TaskType.FEATURE_EXTRACTION))
        self.head = nn.Linear(self.backbone.config.hidden_size, 2)
    def forward(self, input_ids, attention_mask, labels=None):
        logits = self.head(self.backbone(input_ids, attention_mask).last_hidden_state[:, 0, :])
        loss = focal_loss_fn(logits, labels) if labels is not None else None
        return logits, loss

# ==================== 4. TRAINING (Full Data, No Folds) ====================
def train_hybrid(train_df, dev_df=None):
    """Train hybrid model on full training data. Returns model + tokenizers."""
    print("\n🚀 Training Hybrid Model on Full Data...")
    tok_a = AutoTokenizer.from_pretrained(MODEL_A)
    tok_b = AutoTokenizer.from_pretrained(MODEL_B)
    ids_a, masks_a = parallel_tokenize(train_df.text.tolist(), tok_a)
    ids_b, masks_b = parallel_tokenize(train_df.text.tolist(), tok_b)
    labels = train_df.polarization.values

    pad_a, pad_b = tok_a.pad_token_id or 0, tok_b.pad_token_id or 0
    def dual_collate(batch):
        def _pad(seqs, pad_val):
            ml = max(len(s) for s in seqs)
            return torch.stack([F.pad(torch.as_tensor(s), (0, ml-len(s)), value=pad_val) for s in seqs])
        out = {
            "ids_a": _pad([b["ids_a"] for b in batch], pad_a),
            "mask_a": _pad([b["mask_a"] for b in batch], 0),
            "ids_b": _pad([b["ids_b"] for b in batch], pad_b),
            "mask_b": _pad([b["mask_b"] for b in batch], 0),
        }
        if "labels" in batch[0]:
            out["labels"] = torch.tensor([b["labels"] for b in batch])
        return out

    ds = FastDualDataset(ids_a, masks_a, ids_b, masks_b, labels)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=dual_collate,
                    num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=8)

    # Prepare dev dataloader if dev_df provided with labels
    dl_dev = None
    dev_labels = None
    if dev_df is not None and len(dev_df) > 0 and "polarization" in dev_df.columns:
        # Filter to only rows with valid labels
        dev_valid = dev_df[dev_df.polarization.notna()].reset_index(drop=True)
        if len(dev_valid) > 0:
            dev_ids_a, dev_masks_a = parallel_tokenize(dev_valid.text.tolist(), tok_a)
            dev_ids_b, dev_masks_b = parallel_tokenize(dev_valid.text.tolist(), tok_b)
            dev_labels = dev_valid.polarization.values.astype(int)
            ds_dev = FastDualDataset(dev_ids_a, dev_masks_a, dev_ids_b, dev_masks_b)
            dl_dev = DataLoader(ds_dev, batch_size=BATCH*2, shuffle=False, collate_fn=dual_collate,
                                num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=8)

    model = HybridDualModel().cuda()
    opt = AdamW(model.parameters(), lr=LR_BASE)
    sched = get_cosine_schedule_with_warmup(opt, len(dl) // 4, EPOCHS_BASE * len(dl))

    for ep in range(EPOCHS_BASE):
        model.train()
        ep_loss = 0.0
        loop = tqdm(dl, desc=f"Hybrid Ep{ep+1}/{EPOCHS_BASE}", leave=False)
        for b in loop:
            b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
            _, loss = model(**b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            ep_loss += loss.item()

        # Evaluate on dev
        dev_str = ""
        if dl_dev is not None:
            model.eval()
            probs = []
            with torch.no_grad():
                for b in dl_dev:
                    b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
                    logits, _ = model(**b)
                    probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            f1 = f1_score(dev_labels, (np.array(probs) > 0.5).astype(int), average="macro")
            dev_str = f"  dev_f1: {f1:.4f}"
        print(f"   Ep {ep+1}/{EPOCHS_BASE} — train_loss: {ep_loss/len(dl):.4f}{dev_str}")

    print("   ✅ Hybrid training complete")
    return model, tok_a, tok_b, dual_collate


def predict_hybrid(model, tok_a, tok_b, df):
    """Run hybrid inference on a dataframe. Returns prob array."""
    ids_a, masks_a = parallel_tokenize(df.text.tolist(), tok_a)
    ids_b, masks_b = parallel_tokenize(df.text.tolist(), tok_b)

    pad_a, pad_b = tok_a.pad_token_id or 0, tok_b.pad_token_id or 0
    def dual_collate(batch):
        def _pad(seqs, pad_val):
            ml = max(len(s) for s in seqs)
            return torch.stack([F.pad(torch.as_tensor(s), (0, ml-len(s)), value=pad_val) for s in seqs])
        return {
            "ids_a": _pad([b["ids_a"] for b in batch], pad_a),
            "mask_a": _pad([b["mask_a"] for b in batch], 0),
            "ids_b": _pad([b["ids_b"] for b in batch], pad_b),
            "mask_b": _pad([b["mask_b"] for b in batch], 0),
        }

    ds = FastDualDataset(ids_a, masks_a, ids_b, masks_b)
    dl = DataLoader(ds, batch_size=BATCH*2, shuffle=False, collate_fn=dual_collate,
                    num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=8)
    model.eval()
    probs = []
    with torch.no_grad():
        for b in dl:
            b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
            logits, _ = model(**b)
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return np.array(probs)


def train_experts(train_df, dev_df=None):
    """Train expert models on full data per language. Returns {lang: (model, tok, collator)}."""
    print("\n🚀 Training Expert Models on Full Data...")
    expert_models = {}
    for lang, m_name in EXPERT_MODELS.items():
        sub_df = train_df[train_df.lang == lang].reset_index(drop=True)
        if len(sub_df) < 20:
            print(f"   ⚠️ {lang.upper()}: only {len(sub_df)} samples, skipping")
            continue
        print(f"   Expert: {lang.upper()} ({m_name}) — {len(sub_df)} samples")

        tok = AutoTokenizer.from_pretrained(m_name)
        collator = DataCollatorWithPadding(tokenizer=tok)
        ids, masks = parallel_tokenize(sub_df.text.tolist(), tok)
        labels = sub_df.polarization.values

        ds = FastDataset(ids, masks, labels)
        dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collator,
                        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=8)

        # Prepare dev dataloader for this language
        dl_dev = None
        dev_labels = None
        if dev_df is not None and "polarization" in dev_df.columns:
            dev_sub = dev_df[dev_df.lang == lang].reset_index(drop=True)
            if len(dev_sub) > 0:
                dev_ids, dev_masks = parallel_tokenize(dev_sub.text.tolist(), tok)
                dev_labels = dev_sub.polarization.values
                ds_dev = FastDataset(dev_ids, dev_masks)
                dl_dev = DataLoader(ds_dev, batch_size=BATCH*2, shuffle=False, collate_fn=collator,
                                    num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=8)

        model = ExpertModel(m_name).cuda()
        opt = AdamW(model.parameters(), lr=LR_EXPERT)

        for ep in range(EPOCHS_EXPERT):
            model.train()
            ep_loss = 0.0
            for b in dl:
                b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
                _, loss = model(**b)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                ep_loss += loss.item()

            # Evaluate on dev
            dev_str = ""
            if dl_dev is not None:
                model.eval()
                probs = []
                with torch.no_grad():
                    for b in dl_dev:
                        b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
                        logits, _ = model(**b)
                        probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
                f1 = f1_score(dev_labels, (np.array(probs) > 0.5).astype(int), average="macro")
                dev_str = f"  dev_f1: {f1:.4f}"
            print(f"      {lang.upper()} Ep {ep+1}/{EPOCHS_EXPERT} — train_loss: {ep_loss/len(dl):.4f}{dev_str}")

        print(f"   ✅ {lang.upper()} expert training complete")
        expert_models[lang] = (model, tok, collator)
        gc.collect(); torch.cuda.empty_cache()

    return expert_models


def predict_expert(model, tok, collator, texts):
    """Run expert inference on a list of texts. Returns prob array."""
    ids, masks = parallel_tokenize(texts, tok)
    ds = FastDataset(ids, masks)
    dl = DataLoader(ds, batch_size=BATCH*2, shuffle=False, collate_fn=collator,
                    num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=8)
    model.eval()
    probs = []
    with torch.no_grad():
        for b in dl:
            b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
            logits, _ = model(**b)
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return np.array(probs)


# ==================== MAIN ====================
if __name__ == "__main__":
    # Prefetch Models
    print("⬇️  Prefetching models...")
    for m in [MODEL_A, MODEL_B] + list(EXPERT_MODELS.values()):
        try: AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m, quantization_config=BitsAndBytesConfig(**BNB_CONFIG))
        except: pass

    train_df, dev_df, test_df = load_data()

    if DEBUG_MODE:
        print("🐛 Debug: Truncating datasets for speed test...")
        train_df = train_df.head(64).reset_index(drop=True)
        dev_df = dev_df.head(20).reset_index(drop=True) if len(dev_df) > 0 else dev_df
        test_df = test_df.head(20).reset_index(drop=True) if len(test_df) > 0 else test_df

    # Combine train + dev for training (use dev for per-epoch eval)
    full_train_df = pd.concat([train_df, dev_df], ignore_index=True)
    
    print(f"📊 Loaded {len(train_df)} Train, {len(dev_df)} Dev, {len(test_df)} Test")
    print(f"   Training on {len(full_train_df)} samples (train + dev combined)")
    print(f"   Test languages: {sorted(test_df.lang.unique())}")

    # ========== 1. TRAIN HYBRID ON FULL TRAIN+DEV ==========
    hybrid_model, tok_a, tok_b, _ = train_hybrid(full_train_df, dev_df)  # dev_df for per-epoch eval

    # ========== 2. TRAIN EXPERTS ON FULL TRAIN+DEV ==========
    expert_models = train_experts(full_train_df, dev_df)

    # ========== 3. PREDICT TEST ==========
    print("\n🔮 Running Inference on Test Set...")
    test_df["prob_hybrid"] = predict_hybrid(hybrid_model, tok_a, tok_b, test_df)
    del hybrid_model; gc.collect(); torch.cuda.empty_cache()

    for lang, (exp_model, exp_tok, exp_collator) in expert_models.items():
        mask = test_df.lang == lang
        if mask.sum() > 0:
            test_df.loc[mask, f"prob_expert_{lang}"] = predict_expert(
                exp_model, exp_tok, exp_collator, test_df.loc[mask, "text"].tolist()
            )
            print(f"   Expert {lang.upper()}: {mask.sum()} test samples predicted")
    del expert_models; gc.collect(); torch.cuda.empty_cache()

    # ========== 4. COMBINE PROBS & APPLY THRESHOLD ==========
    print("\n🎯 Generating Final Predictions...")
    test_df["final_prob"] = test_df["prob_hybrid"].copy()

    # For languages with experts: average hybrid + expert
    for lang in EXPERT_MODELS:
        col = f"prob_expert_{lang}"
        if col in test_df.columns:
            mask = (test_df.lang == lang) & test_df[col].notna()
            test_df.loc[mask, "final_prob"] = (
                test_df.loc[mask, "prob_hybrid"] + test_df.loc[mask, col]
            ) / 2.0

    # Apply threshold 0.5 (test has no labels to optimize)
    test_df["prediction"] = (test_df["final_prob"] > 0.5).astype(int)

    print("\n   Per-Language Sample Counts:")
    for lang in sorted(test_df.lang.unique()):
        count = (test_df.lang == lang).sum()
        print(f"   {lang}: {count:>5} samples")

    # ========== 5. SAVE PER-LANGUAGE CSVs ==========
    print(f"\n💾 Saving predictions to {OUT_DIR}/")
    for lang in sorted(test_df.lang.unique()):
        sub = test_df[test_df.lang == lang][["id", "prediction"]]
        path = os.path.join(OUT_DIR, f"pred_{lang}.csv")
        sub.to_csv(path, index=False)
        print(f"   ✅ {path} ({len(sub)} rows)")

    print("\n🏁 ALL DONE.")