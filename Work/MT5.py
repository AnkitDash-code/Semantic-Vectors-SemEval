# ============================================================
# SemEval Subtask 1 — MT5-Large (4-bit QLoRA + Cosine)
# ============================================================

import os, glob, random, gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    get_cosine_schedule_with_warmup, 
    Adafactor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from tqdm import tqdm

# ---------- CONFIG ----------
MODEL_NAME = "google/mt5-large"
BASE_DIR = "/kaggle/input/subtask1/subtask1" # Update if needed
TRAIN_DIR = f"{BASE_DIR}/train"
DEV_DIR = f"{BASE_DIR}/dev"
PRED_DIR = "/kaggle/working/predictions"
os.makedirs(PRED_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
MAX_LEN = 128
# MT5-Large is 1.2B. In 4-bit, it takes ~1.5GB VRAM.
# We can use a healthy batch size.
BATCH_SIZE = 16 
ACCUM_STEPS = 2
EPOCHS = 3
LR = 1e-3 
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- LANG CLUSTERS ----------
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
CLUSTER2ID = {c:i for i,c in enumerate(LANG_CLUSTERS)}

# ---------- DATA ----------
def load_train():
    dfs=[]
    files = glob.glob(f"{TRAIN_DIR}/*.csv")
    if not files: raise ValueError(f"No CSVs in {TRAIN_DIR}")
    for f in files:
        lang=os.path.basename(f).split(".")[0]
        df=pd.read_csv(f)
        df["lang"]=lang
        dfs.append(df)
    df=pd.concat(dfs)
    strat=df.polarization.astype(str)+"_"+df.lang
    return train_test_split(df,test_size=0.15,stratify=strat,random_state=SEED)

def load_dev():
    data={}
    for f in glob.glob(f"{DEV_DIR}/*.csv"):
        lang=os.path.basename(f).split(".")[0]
        df=pd.read_csv(f)
        df["lang"]=lang
        data[lang]=df
    return data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df.text.tolist()
        self.labels = df.polarization.tolist()
        self.tokenizer = tokenizer
        self.prompts = [f"classify polarization: {t}" for t in self.texts]

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        prompt = self.prompts[i]
        target_text = "positive" if self.labels[i] == 1 else "negative"
        
        # Inputs
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True, 
            max_length=MAX_LEN
        )
        
        # Targets
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                target_text,
                return_tensors="pt",
                padding="max_length",
                max_length=8 
            )
        
        input_ids = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        labels = targets.input_ids.squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return input_ids, attention_mask, labels, self.labels[i]

# ---------- COLLATE ----------
def collate(b):
    input_ids = torch.stack([x[0] for x in b])
    attention_mask = torch.stack([x[1] for x in b])
    labels = torch.stack([x[2] for x in b])
    orig_labels = torch.tensor([x[3] for x in b])
    return input_ids, attention_mask, labels, orig_labels

# ---------- MODEL SETUP (4-Bit QLoRA) ----------
def get_model():
    print(f"🚀 Loading {MODEL_NAME} (4-bit QLoRA)...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16, # T4 friendly
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # Load Model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=DEVICE
    )
    
    # Stabilize for 4-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA Config (MT5/T5 Standard Targets)
    peft_config = LoraConfig(
        r=16, lora_alpha=32, 
        target_modules=["q", "v", "k", "o", "wi", "wo"], # Standard T5 names
        lora_dropout=0.05, 
        bias="none", 
        task_type=TaskType.SEQ_2_SEQ_LM 
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer

# ---------- HELPER: PROBABILITIES ----------
def get_binary_probs(model, tokenizer, input_ids, attention_mask):
    """
    Extracts probability of 'positive' vs 'negative' generation.
    """
    pos_id = tokenizer.encode("positive", add_special_tokens=False)[0]
    neg_id = tokenizer.encode("negative", add_special_tokens=False)[0]
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True
        )
    
    first_token_logits = outputs.scores[0]
    pos_logits = first_token_logits[:, pos_id]
    neg_logits = first_token_logits[:, neg_id]
    
    stacked_logits = torch.stack([neg_logits, pos_logits], dim=1)
    probs = torch.softmax(stacked_logits, dim=1)
    return probs[:, 1].cpu().tolist()

# ---------- BASELINE TEST (INITIAL EVAL) ----------
def test_baseline(model, tokenizer, val_loader):
    print("\n🔍 Running Baseline Evaluation (Pre-Train)...")
    model.eval()
    probs, refs = [], []
    
    # Quick check on a subset (2 batches) to verify stability
    limit = 2 * BATCH_SIZE
    count = 0
    
    for input_ids, attention_mask, labels, orig_labels in tqdm(val_loader, desc="Baseline"):
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        
        batch_probs = get_binary_probs(model, tokenizer, input_ids, attention_mask)
        probs.extend(batch_probs)
        refs.extend(orig_labels.tolist())
        
        count += len(input_ids)
        if count >= limit: break

    preds = (np.array(probs) > 0.5).astype(int)
    score = f1_score(refs, preds, average="macro")
    print(f"📊 Baseline F1 (Subset): {score:.4f}\n")

# ---------- TRAIN ----------
def train():
    gc.collect()
    torch.cuda.empty_cache()

    train_df, val_df = load_train()
    model, tokenizer = get_model()
    
    tl = DataLoader(Dataset(train_df, tokenizer), batch_size=BATCH_SIZE, shuffle=True, 
                    collate_fn=collate, num_workers=2)
    vl = DataLoader(Dataset(val_df, tokenizer), batch_size=BATCH_SIZE, 
                    collate_fn=collate, num_workers=2)

    # 1. Initial Eval
    test_baseline(model, tokenizer, vl)

    # 2. Optimizer & Scheduler
    # Adafactor is safer for T5-family models in low precision
    opt = Adafactor(model.parameters(), lr=LR, relative_step=False, scale_parameter=False)
    # Cosine Scheduler (The specific change you wanted)
    sch = get_cosine_schedule_with_warmup(opt, 100, EPOCHS * len(tl) // ACCUM_STEPS)
    
    best_score = 0.0
    best_path = f"{PRED_DIR}/best_model.pt"

    print(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for e in range(EPOCHS):
        model.train()
        loop = tqdm(tl, desc=f"Epoch {e+1}")
        
        for i, (input_ids, attention_mask, labels, _) in enumerate(loop):
            input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / ACCUM_STEPS
            loss.backward()
            
            if (i+1) % ACCUM_STEPS == 0:
                opt.step()
                sch.step()
                opt.zero_grad()
            
            loop.set_postfix(loss=loss.item() * ACCUM_STEPS)

        # Validation
        model.eval()
        probs, refs = [], []
        print("🔍 Validating...")
        
        for input_ids, attention_mask, labels, orig_labels in tqdm(vl):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            batch_probs = get_binary_probs(model, tokenizer, input_ids, attention_mask)
            probs.extend(batch_probs)
            refs.extend(orig_labels.tolist())

        preds = (np.array(probs) > 0.5).astype(int)
        score = f1_score(refs, preds, average="macro")
        print(f"🔥 Epoch {e+1} F1 (Th=0.5): {score:.4f}")
        
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)
            print(f"✅ Saved Best: {score:.4f}")

    print("↺ Loading Best Model...")
    model.load_state_dict(torch.load(best_path))
    
    # Final Probs for Tuning
    model.eval()
    final_probs = []
    for input_ids, attention_mask, labels, orig_labels in tqdm(vl, desc="Final Eval"):
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        batch_probs = get_binary_probs(model, tokenizer, input_ids, attention_mask)
        final_probs.extend(batch_probs)

    val_df["prob"] = final_probs
    return model, tokenizer, val_df

# ---------- THRESHOLD TUNING ----------
def tune_cluster_thresholds(val):
    print("\n⚖️ Tuning Thresholds...")
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

# ---------- PREDICT ----------
def predict(model, tokenizer, th, val_stats):
    print("\n📝 Predicting...")
    dev = load_dev()
    model.eval()
    lang_stats = val_stats.groupby("lang")["prob"].agg(["mean", "std"]).to_dict("index")
    
    for lang, df in dev.items():
        ds = Dataset(df, tokenizer)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate, num_workers=2)
        
        probs = []
        for input_ids, attention_mask, _, _ in tqdm(dl, desc=lang):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            batch_probs = get_binary_probs(model, tokenizer, input_ids, attention_mask)
            probs.extend(batch_probs)
            
        stats = lang_stats.get(lang, {"mean": 0.5, "std": 0.1})
        cluster_th = th[LANG2CLUSTER[lang]]
        norm_probs = (np.array(probs) - stats["mean"]) / (stats["std"] + 1e-9)
        preds = (norm_probs > cluster_th).astype(int)
        
        pd.DataFrame({"id": df.id, "polarization": preds}).to_csv(f"{PRED_DIR}/pred_{lang}.csv", index=False)
    
    print(f"✅ All Done!")

# ---------- RUN PIPELINE ----------
model, tokenizer, val_df = train()
thresholds, val_df_with_stats = tune_cluster_thresholds(val_df)
predict(model, tokenizer, thresholds, val_df_with_stats)