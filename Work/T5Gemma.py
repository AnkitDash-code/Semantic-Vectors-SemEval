# ============================================================
# SemEval Subtask 1 — T5-Gemma-2 (Final Stable & Optimized)
# ============================================================

import os, glob, random
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_cosine_schedule_with_warmup, # Changed to Cosine for smooth landing
    Adafactor
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# ---------- CONFIG ----------
MODEL_NAME = "google/t5gemma-2-1b-1b"
# UPDATE THIS PATH if your dataset location is different
BASE_DIR = "/kaggle/input/subtask1/subtask1" 
TRAIN_DIR = f"{BASE_DIR}/train"
DEV_DIR = f"{BASE_DIR}/dev"
PRED_DIR = "/kaggle/working/predictions"
os.makedirs(PRED_DIR, exist_ok=True)

# Optimized for Single T4 (FP32 Mode)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_LEN = 128
# FP32 takes more RAM. Batch 16 fits safely. 
# We accum 2 steps to simulate Batch 32 for stability.
BATCH_SIZE = 16 
ACCUM_STEPS = 2 
EPOCHS = 3
LR = 1e-3  # Adafactor default
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
    # CHECK PATHS if this fails
    files = glob.glob(f"{TRAIN_DIR}/*.csv")
    if not files:
        raise ValueError(f"No CSV files found in {TRAIN_DIR}. Check BASE_DIR path!")
        
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
    def __init__(self,df):
        self.t=df.text.tolist()
        self.l=df.polarization.tolist()
        self.lang=df.lang.tolist()
    def __len__(self): return len(self.t)
    def __getitem__(self,i): return self.t[i],self.l[i],self.lang[i]

# ---------- MODEL (FP32 Stable + Fixed Targets) ----------
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Load Full Model in FP32 (Default for stability)
        full_model = AutoModel.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True
        )
        
        # 2. Extract Encoder
        if hasattr(full_model, "get_encoder"):
            self.enc = full_model.get_encoder()
        elif hasattr(full_model, "encoder"):
            self.enc = full_model.encoder
        else:
            self.enc = full_model 
            
        if hasattr(self.enc, "gradient_checkpointing_enable"):
            self.enc.gradient_checkpointing_enable()

        # 3. LoRA Config (Gemma Targets)
        # Increased dropout to 0.1 for regularization
        peft_config = LoraConfig(
            r=32, lora_alpha=64, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
            lora_dropout=0.1,  # STABILIZER: Increased dropout
            bias="none", 
            task_type=TaskType.FEATURE_EXTRACTION
        )
        self.enc = get_peft_model(self.enc, peft_config)
        
        # 4. Get Hidden Size (Robust Check)
        dim = None
        for attr in ["d_model", "hidden_size", "n_embd", "embed_dim"]:
            if hasattr(self.enc.config, attr):
                dim = getattr(self.enc.config, attr)
                break
        if dim is None: dim = 1152 # Fallback for 1B model
            
        print(f"✅ Detected Model Hidden Dimension: {dim}")

        # 5. Heads
        self.norm = nn.LayerNorm(dim)
        self.cluster = nn.Embedding(len(CLUSTER2ID), 32)
        self.proj = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(), 
            nn.Linear(512, 128)
        )
        self.clf = nn.Linear(dim + 32, 2)

    def mean_pool(self,x,m):
        m=m.unsqueeze(-1)
        return (x*m).sum(1)/m.sum(1).clamp(min=1e-9)

    def forward(self,ids,mask,cid):
        out = self.enc(ids, attention_mask=mask).last_hidden_state
        emb = self.mean_pool(out, mask)
        emb = self.norm(emb)
        cluster_emb = self.cluster(cid)
        return self.clf(torch.cat([emb, cluster_emb], 1)), self.proj(emb)

# ---------- SUPCON ----------
class SupCon(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, y):
        z = nn.functional.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        labels = y.unsqueeze(1)
        mask = (labels == labels.T).float().to(z.device)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0)).to(z.device)
        mask = mask * logits_mask
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
        loss = -(mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        return loss.mean()

# ---------- COLLATE ----------
def collate(b,tk):
    t,l,lang=zip(*b)
    cid=torch.tensor([CLUSTER2ID[LANG2CLUSTER[x]] for x in lang])
    enc=tk(list(t),padding=True,truncation=True,max_length=MAX_LEN,return_tensors="pt")
    return enc["input_ids"],enc["attention_mask"],torch.tensor(l),cid

# ---------- BASELINE TEST ----------
def test_baseline(model, val_loader):
    print("\n🔍 Running Pre-Train Baseline Test (T5-Gemma-2 FP32)...")
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for ids, mask, y, c in tqdm(val_loader, desc="Baseline Eval"):
            ids, mask, c = ids.to(DEVICE), mask.to(DEVICE), c.to(DEVICE)
            logits, _ = model(ids, mask, c)
            prob = torch.softmax(logits, 1)[:, 1].cpu().tolist()
            yp += prob
            yt += y.tolist()
    overall_f1 = f1_score(yt, (np.array(yp) > 0.5).astype(int), average="macro")
    print(f"\n📊 BASELINE Overall Macro-F1: {overall_f1:.4f}\n")

# ---------- TRAIN ----------
def train():
    train_df,val_df=load_train()
    # Trust remote code needed for new models
    tk=AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    tl = DataLoader(Dataset(train_df), batch_size=BATCH_SIZE, shuffle=True, 
                    collate_fn=lambda x:collate(x,tk), num_workers=4, pin_memory=True)
    vl = DataLoader(Dataset(val_df), batch_size=BATCH_SIZE, 
                    collate_fn=lambda x:collate(x,tk), num_workers=4, pin_memory=True)

    model = Model().to(DEVICE)
    con = SupCon()

    test_baseline(model, vl)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # STABILIZER 1: Weight Decay to prevent overfitting
    opt = Adafactor(
        trainable_params,
        lr=LR,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        weight_decay=0.01, # Added Regularization
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
    )
    
    # STABILIZER 2: Cosine Scheduler for smooth landing
    sch = get_cosine_schedule_with_warmup(opt, 200, EPOCHS*len(tl)//ACCUM_STEPS)
    
    # STABILIZER 3: Save Best Model logic
    best_score = 0.0
    best_model_path = f"{PRED_DIR}/best_model.pt"

    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for e in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loop = tqdm(tl, desc=f"Epoch {e+1}")
        
        for i,(ids,mask,y,c) in enumerate(loop):
            ids,mask,y,c=ids.to(DEVICE),mask.to(DEVICE),y.to(DEVICE),c.to(DEVICE)
            
            # Forward (Implicit FP32)
            logits,z=model(ids,mask,c)
            loss=(nn.functional.cross_entropy(logits,y)+0.2*con(z,y))/ACCUM_STEPS
            
            loss.backward()
            
            if (i+1)%ACCUM_STEPS==0:
                opt.step()
                sch.step()
                opt.zero_grad()
            
            loop.set_postfix(loss=loss.item()*ACCUM_STEPS)

        # Validation Step
        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for ids,mask,y,c in vl:
                ids,mask,c=ids.to(DEVICE),mask.to(DEVICE),c.to(DEVICE)
                logits,_=model(ids,mask,c)
                prob=torch.softmax(logits,1)[:,1].cpu().tolist()
                yp+=prob
                yt+=y.tolist()

        score = f1_score(yt,(np.array(yp)>0.5).astype(int),average="macro")
        print(f"\n🔥 Epoch {e+1} Macro-F1: {score:.4f}")

        # Check for improvement
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New Best Model Saved! (Score: {best_score:.4f})")
        else:
            print(f"⚠️ No improvement (Best: {best_score:.4f})")
            
    # LOAD BEST AT END
    print(f"\n↺ Loading Best Model (Score: {best_score:.4f}) for Predictions...")
    model.load_state_dict(torch.load(best_model_path))

    val_df = val_df.reset_index(drop=True)
    val_df["prob"] = yp
    return model,tk,val_df

# ---------- THRESHOLD TUNING ----------
def tune_cluster_thresholds(val):
    print("\n⚖️ Tuning Thresholds per Cluster...")
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
def predict(model, tk, th, val_stats):
    print("\n📝 Generating Predictions...")
    dev = load_dev()
    model.eval()
    lang_stats = val_stats.groupby("lang")["prob"].agg(["mean", "std"]).to_dict("index")
    
    for lang, df in dev.items():
        probs = []
        for txt in tqdm(df.text, desc=lang):
            enc = tk(txt, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            ids = enc["input_ids"].to(DEVICE)
            mask = enc["attention_mask"].to(DEVICE)
            cid = torch.tensor([CLUSTER2ID[LANG2CLUSTER[lang]]]).to(DEVICE)
            with torch.no_grad():
                logits, _ = model(ids, mask, cid)
                p = torch.softmax(logits, 1)[0, 1].item()
                probs.append(p)
        
        stats = lang_stats.get(lang, {"mean": 0.5, "std": 0.1}) 
        norm_probs = (np.array(probs) - stats["mean"]) / (stats["std"] + 1e-9)
        cluster_th = th[LANG2CLUSTER[lang]]
        preds = (norm_probs > cluster_th).astype(int)
        
        pd.DataFrame({"id": df.id, "polarization": preds}).to_csv(f"{PRED_DIR}/pred_{lang}.csv", index=False)
    print(f"✅ All predictions saved to {PRED_DIR}")

# ---------- RUN PIPELINE ----------
model, tk, val_df = train()
thresholds, val_df_with_stats = tune_cluster_thresholds(val_df)
predict(model, tk, thresholds, val_df_with_stats)