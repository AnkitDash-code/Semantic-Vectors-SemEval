# ==================================================================================
# SemEval SOTA FINAL: mDeBERTa QLoRA + Smart Batching + MSD + CV + Tuning + TTA
# FIXED: strict=False added to load_state_dict to handle quantization keys
# ==================================================================================

import os, glob, random, gc, time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Sampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.optim import AdamW 
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    DataCollatorWithPadding, 
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Helps with memory fragmentation on T4
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------- CONFIG ----------
MODEL_NAME = "microsoft/mdeberta-v3-base" 

BASE_DIR = "/kaggle/input/subtask1/subtask1" 
TRAIN_DIR = f"{BASE_DIR}/train"
DEV_DIR = f"{BASE_DIR}/dev"
PRED_DIR = "/kaggle/working/predictions"
os.makedirs(PRED_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# HYPERPARAMETERS
BATCH_SIZE = 32       
ACCUM_STEPS = 1       
EPOCHS = 5
LR = 2e-4             # QLoRA requires higher LR
RANK = 64             
ALPHA = 128
SEED = 42
N_FOLDS = 4           # Stratified CV

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- SMART DATA ENGINE ----------
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, is_test=False):
        self.texts = df.text.tolist()
        self.labels = df.polarization.tolist() if not is_test else [0]*len(df)
        self.tokenizer = tokenizer
        # Store lengths for Smart Batching sorting
        self.lengths = [len(t) for t in self.texts] 
        self.ids = df.id.tolist() if is_test else None
        
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): 
        return {"text": self.texts[i], "label": self.labels[i]}

class SmartBatchSampler(Sampler):
    """Sorts data by length to minimize padding overhead (Speedup ~3x)."""
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.sorted_indices = np.argsort(self.data_source.lengths)
        
    def __iter__(self):
        # Create batches from sorted indices
        batches = []
        for i in range(0, len(self.sorted_indices), self.batch_size):
            batches.append(self.sorted_indices[i:i+self.batch_size])
        # Important: Shuffle the batches themselves for stochastic training
        random.shuffle(batches) 
        for batch in batches: yield from batch
        
    def __len__(self): return len(self.data_source)

def custom_collate(batch):
    tokenizer = globals()['tokenizer_ref']
    texts = [b['text'] for b in batch]
    labels = torch.tensor([b['label'] for b in batch])
    # Dynamic Padding: Pad to the longest sequence in THIS BATCH only
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return inputs.input_ids, inputs.attention_mask, labels

# ---------- MODEL: 4-BIT QLORA + MSD ----------
class QLoraDebertaModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. 4-bit Config (Compute in FP32 to avoid Overflow)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float32 # CRITICAL: FP32 Compute prevents overflow
        )
        
        # 2. Backbone
        self.backbone = AutoModel.from_pretrained(
            MODEL_NAME, 
            quantization_config=bnb_config,
            device_map=DEVICE
        )
        self.backbone = prepare_model_for_kbit_training(self.backbone)
        
        # 3. LoRA (Target ALL Linear Layers)
        peft_config = LoraConfig(
            r=RANK,
            lora_alpha=ALPHA,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        self.backbone = get_peft_model(self.backbone, peft_config)
        
        # 4. MSD Head (FP32)
        self.config = AutoConfig.from_pretrained(MODEL_NAME)
        # 5 Parallel Dropouts for robustness
        self.dropouts = nn.ModuleList([nn.Dropout(0.1 * (i+1)) for i in range(5)])
        self.fc = nn.Linear(self.config.hidden_size, 2).to(DEVICE)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        
        # Mean Pooling (Standard for DeBERTa)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Multi-Sample Dropout Forward
        logits_list = []
        for dropout in self.dropouts:
            logits_list.append(self.fc(dropout(embeddings)))
        # Average the predictions
        logits = torch.mean(torch.stack(logits_list), dim=0)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Average the loss from all dropout samples
            losses = [loss_fct(l.view(-1, 2), labels.view(-1)) for l in logits_list]
            loss = torch.mean(torch.stack(losses))
            
        return type('Output', (object,), {'loss': loss, 'logits': logits})

# ---------- STEP 1: TRAIN (CROSS-VALIDATION) ----------
def run_cv_training():
    dfs=[]
    for f in glob.glob(f"{TRAIN_DIR}/*.csv"):
        lang=os.path.basename(f).split(".")[0]
        df=pd.read_csv(f)
        df["lang"]=lang
        dfs.append(df)
    full_df = pd.concat(dfs).reset_index(drop=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    globals()['tokenizer_ref'] = tokenizer
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    strat_col = full_df.polarization.astype(str) + "_" + full_df.lang
    
    oof_df = full_df.copy()
    oof_df['prob'] = 0.0
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, strat_col)):
        print(f"\n🔄 Fold {fold+1}/{N_FOLDS}")
        
        train_sub = full_df.iloc[train_idx].reset_index(drop=True)
        val_sub = full_df.iloc[val_idx].reset_index(drop=True)
        
        train_ds = TextDataset(train_sub, tokenizer)
        val_ds = TextDataset(val_sub, tokenizer)
        
        # Use Smart Batching for Training speed
        train_sampler = SmartBatchSampler(train_ds, BATCH_SIZE)
        tl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=custom_collate, num_workers=2)
        vl = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, collate_fn=custom_collate, num_workers=2)
        
        print(f"🚀 Initializing QLoRA Model...")
        model = QLoraDebertaModel()
        
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=EPOCHS*len(tl))
        # Note: No GradScaler needed for FP32 compute path
        
        best_fold_score = 0.0
        best_path = f"{PRED_DIR}/model_fold_{fold}.pt"
        
        for e in range(EPOCHS):
            model.train()
            loop = tqdm(tl, desc=f"Ep {e+1}")
            for i, (input_ids, mask, labels) in enumerate(loop):
                input_ids, mask, labels = input_ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(input_ids, mask, labels)
                loss = outputs.loss / ACCUM_STEPS
                
                loss.backward()
                
                if (i+1) % ACCUM_STEPS == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    loop.set_postfix(loss=loss.item() * ACCUM_STEPS)
            
            # Validation
            model.eval()
            probs, refs = [], []
            with torch.no_grad():
                for input_ids, mask, labels in vl:
                    input_ids, mask = input_ids.to(DEVICE), mask.to(DEVICE)
                    outputs = model(input_ids, mask)
                    probs.extend(torch.softmax(outputs.logits, dim=1)[:, 1].cpu().tolist())
                    refs.extend(labels.tolist())
            
            score = f1_score(refs, (np.array(probs) > 0.5).astype(int), average="macro")
            print(f"   >>> F1: {score:.4f}")
            if score > best_fold_score:
                best_fold_score = score
                torch.save(model.state_dict(), best_path)
                
        # --- FIX: Load Best Model with strict=False ---
        print("↺ Loading Best Model (Strict=False)...")
        model.load_state_dict(torch.load(best_path), strict=False)
        
        model.eval()
        oof_probs = []
        with torch.no_grad():
            for input_ids, mask, labels in vl:
                input_ids, mask = input_ids.to(DEVICE), mask.to(DEVICE)
                outputs = model(input_ids, mask)
                oof_probs.extend(torch.softmax(outputs.logits, dim=1)[:, 1].cpu().tolist())
        
        oof_df.loc[val_idx, 'prob'] = oof_probs
        scores.append(best_fold_score)
        
        del model, optimizer, scheduler; gc.collect(); torch.cuda.empty_cache()
    
    print(f"\n🏆 Average CV Score: {np.mean(scores):.4f}")
    return tokenizer, oof_df

# ---------- STEP 2: TUNE THRESHOLDS ----------
def tune_thresholds(val):
    print("\n⚖️ Tuning Thresholds...")
    LANG_CLUSTERS = {
        "western": ["eng","deu","ita","spa","pol"],
        "indic": ["hin","ben","tel","ori","pan","urd","nep"],
        "semitic": ["arb","fas"],
        "african": ["amh","hau","swa"],
        "southeast_asia": ["mya","khm"],
        "sinitic": ["zho"],
        "turkic": ["tur"],
        "slavic": ["rus"],
    }
    LANG2CLUSTER = {l:c for c,ls in LANG_CLUSTERS.items() for l in ls}
    
    val["cluster"] = val.lang.map(lambda x: LANG2CLUSTER[x])
    val["norm_prob"] = val.groupby("lang")["prob"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )

    th = {}
    for c in val.cluster.unique():
        best_t, best_f1 = 0, 0
        sub = val[val.cluster == c]
        for t in np.arange(-2.0, 2.0, 0.05):
            preds = (sub.norm_prob > t).astype(int)
            f = f1_score(sub.polarization, preds, average="macro")
            if f > best_f1: best_f1 = f; best_t = t
        th[c] = best_t
        print(f"Cluster: {c:<15} | Threshold: {best_t:.2f} | F1: {best_f1:.4f}")
    return th, val

# ---------- STEP 3: PREDICT (TTA + ENSEMBLE) ----------
def predict_final(tokenizer, th, val_stats):
    print("\n📝 Predicting with TTA + Ensemble...")
    lang_stats = val_stats.groupby("lang")["prob"].agg(["mean", "std"]).to_dict("index")
    
    LANG_CLUSTERS = {
        "western": ["eng","deu","ita","spa","pol"],
        "indic": ["hin","ben","tel","ori","pan","urd","nep"],
        "semitic": ["arb","fas"],
        "african": ["amh","hau","swa"],
        "southeast_asia": ["mya","khm"],
        "sinitic": ["zho"],
        "turkic": ["tur"],
        "slavic": ["rus"],
    }
    LANG2CLUSTER = {l:c for c,ls in LANG_CLUSTERS.items() for l in ls}
    
    dev_data = {}
    for f in glob.glob(f"{DEV_DIR}/*.csv"):
        dev_data[os.path.basename(f).split(".")[0]] = pd.read_csv(f)

    for lang, df in dev_data.items():
        # Prepare Data (Original + Lowercase)
        ds_orig = TextDataset(df, tokenizer, is_test=True)
        ds_lower = TextDataset(df.copy(), tokenizer, is_test=True)
        ds_lower.texts = [t.lower() for t in ds_lower.texts]
        
        dl_orig = DataLoader(ds_orig, batch_size=32, shuffle=False, collate_fn=custom_collate, num_workers=2)
        dl_lower = DataLoader(ds_lower, batch_size=32, shuffle=False, collate_fn=custom_collate, num_workers=2)
        
        final_probs = np.zeros(len(df))
        
        # Loop through Folds
        for fold in range(N_FOLDS):
            model = QLoraDebertaModel()
            # --- FIX: Load with strict=False here too ---
            model.load_state_dict(torch.load(f"{PRED_DIR}/model_fold_{fold}.pt"), strict=False)
            model.eval()
            
            # Predict Original
            p_orig = []
            with torch.no_grad():
                for input_ids, mask, _ in dl_orig:
                    input_ids, mask = input_ids.to(DEVICE), mask.to(DEVICE)
                    p_orig.extend(torch.softmax(model(input_ids, mask).logits, dim=1)[:, 1].cpu().tolist())
            
            # Predict Lowercase (TTA)
            p_lower = []
            with torch.no_grad():
                for input_ids, mask, _ in dl_lower:
                    input_ids, mask = input_ids.to(DEVICE), mask.to(DEVICE)
                    p_lower.extend(torch.softmax(model(input_ids, mask).logits, dim=1)[:, 1].cpu().tolist())
            
            # Ensemble + TTA Average
            final_probs += (np.array(p_orig) + np.array(p_lower)) / 2.0
            del model; gc.collect(); torch.cuda.empty_cache()
            
        final_probs /= N_FOLDS
        
        # Apply Tuned Thresholds
        stats = lang_stats.get(lang, {"mean": 0.5, "std": 0.1})
        cluster_th = th[LANG2CLUSTER[lang]]
        norm_probs = (final_probs - stats["mean"]) / (stats["std"] + 1e-9)
        preds = (norm_probs > cluster_th).astype(int)
        
        pd.DataFrame({"id": df.id, "polarization": preds}).to_csv(f"{PRED_DIR}/pred_{lang}.csv", index=False)
    
    print(f"✅ All Done! Predictions saved to {PRED_DIR}")

# ---------- EXECUTE ALL ----------
tokenizer, oof_df = run_cv_training()
thresholds, oof_stats = tune_thresholds(oof_df)
predict_final(tokenizer, thresholds, oof_stats)