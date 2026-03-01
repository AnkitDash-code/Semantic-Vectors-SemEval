# Semantic Vectors at SemEval-2026 Task 9: Robust Multilingual Polarization Detection via Dual-Encoder Fusion and Expert Ensembling

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.40.0-orange.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

This repository contains the **SemanticVectors** system for **POLAR@SemEval-2026 Task 9: Subtask 1 — Binary Polarization Detection** across 22 typologically diverse languages. Online polarization is frequently conveyed through implicit rhetorical framing (irony, dog-whistles, rhetorical questions), making it far harder to detect than explicit hate speech. Our system addresses this with a four-stage pipeline: Siamese dual-encoder training, language-specific expert distillation, XGBoost meta-stacking with Platt calibration, and per-language threshold optimization.

📄 **Research Paper**: Full methodology and experimental analysis are documented in [`Papers/semeval_mine_v5.tex`](Papers/semeval_mine_v5.tex).

**Authors**: Priyanshu Mittal, Ankit Dash, Piyush Prashant — Indian Institute of Information Technology Dharwad

### Key Achievements

- 🏆 **Macro-F1: 0.797** · **Accuracy: 0.827** across all 22 languages on the hidden test set
- 🌍 **22 Languages** spanning 8 typological families (Western European, Indic, Semitic, African, Southeast Asian, Sinitic, Turkic, Slavic)
- 🧠 **Siamese Dual-Encoder** — mDeBERTa-v3-large + XLM-RoBERTa-large via 4-bit QLoRA (+1.8 pp over single encoder)
- 📊 **XGBoost Meta-Stacker** with Shannon entropy features and per-language Platt calibration
- 🎯 **Focal Loss as Hard-Example Miner** — concentrates gradients on subtly framed rhetorical instances (γ=2.0 down-weights easy examples by up to 96%)
- 🔢 **Per-Language Threshold Optimization** via 81-point grid search on development data

---

## 📁 Project Structure

```
Work/
│
├── Notebooks/  ──  Exploration & Prototype Phases
│   ├── Semeval.ipynb                          # Phase 1 — BERT + BitNet 1.58-bit (English only baseline)
│   ├── Semeval_multilingual.ipynb             # Phase 2 — 9-language mDeBERTa BitNet extension
│   ├── Semeval_Optimized.ipynb                # Phase 2.1 — Threshold grid-search & hyperparameter tuning
│   ├── Semeval_multilingual_IMPROVED.ipynb    # Phase 3 — Data augmentation + LoRA language adapters
│   ├── SemEval_RWK.ipynb                      # Phase 4 — RWKV O(N) efficient architecture (negative result)
│   ├── semeval_mamba.ipynb                    # Phase 5 — Mamba state-space model (experimental)
│   ├── Semantic_mutilingual_deberta.ipynb     # Phase 6 — First production XLM-RoBERTa training pipeline
│   └── Roberta_model2.ipynb                   # Phase 6.1 — RoBERTa ablation experiments
│
├── Scripts/  ──  Model Scaling & Advanced Tuning
│   ├── new_model.py                           # Baseline script migrated from notebooks
│   ├── roberta_large.py                       # RoBERTa Large single-model baseline
│   ├── MT5.py                                 # Sequence-to-sequence encoder-decoder approach
│   ├── T5Gemma.py                             # Generative Gemma-based classification
│   ├── MDeberta-XLM.py                        # mDeBERTa + XLM-R initial dual-model integration
│   ├── Mdeberta-XLM-final.py                  # Refined mDeBERTa + XLM-R with tuned fusion
│   └── Mdeberta-QLora.py                      # QLoRA 4-bit fine-tuning for memory-efficient training
│
├── Final Submissions/
│   ├── lightning.py                           # Phase 8 — PyTorch Lightning migration (2nd-last version)
│   ├── final_submission_XLM-Mdeberta-Expert.py# Phase 9 — FINAL: Siamese encoder + expert ensembling
│   └── Final_Code_Paper_Submission.py         # Phase 10 — Paper-ready reproducible pipeline (WIP)
│
├── Papers/
│   └── semeval_mine_v5.tex                    # Full ACL-format system description paper
│
├── dev_phase_data/subtask1/                   # Dev-phase dataset (9 languages)
│   ├── train/                                 # Per-language CSVs (amh, arb, deu, eng, hau, ita, spa, urd, zho)
│   └── dev/
│
├── Work/subtask1/                             # Full competition dataset (22 languages)
│   ├── train/                                 # Per-language CSVs
│   └── dev/
│
├── Work/predictions_qwen3/                    # Inference outputs from external Qwen3 model
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AnkitDash-code/Semantic-Vectors-SemEval
cd Semantic-Vectors-SemEval

# Install dependencies
pip install torch>=2.0.0 transformers>=4.40.0 peft accelerate xgboost scikit-learn pandas numpy bitsandbytes
```

### Run the Final System (Siamese Dual-Encoder + Expert Ensemble)

```python
# final_submission_XLM-Mdeberta-Expert.py
# Loads both mDeBERTa-v3-large and XLM-RoBERTa-large with 4-bit QLoRA,
# runs the XGBoost meta-stacker, and outputs per-language predictions.

python Work/final_submission_XLM-Mdeberta-Expert.py \
    --train_dir Work/subtask1/train/ \
    --dev_dir   Work/subtask1/dev/ \
    --output_dir predictions/
```

### Quick Inference with the Dual Encoder

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load both backbone tokenizers
tok_deb  = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
tok_xlmr = AutoTokenizer.from_pretrained("xlm-roberta-large")

def encode_dual(text, model_deb, model_xlmr, max_len=256):
    """
    Encodes text through both encoders and concatenates [CLS] vectors
    to form a 2048-dimensional joint representation.
    """
    enc_d = tok_deb(text,  return_tensors="pt", truncation=True, max_length=max_len, padding=True)
    enc_x = tok_xlmr(text, return_tensors="pt", truncation=True, max_length=max_len, padding=True)

    with torch.no_grad():
        h_deb  = model_deb(**enc_d).last_hidden_state[:, 0, :]   # [CLS] → R^1024
        h_xlmr = model_xlmr(**enc_x).last_hidden_state[:, 0, :]  # [CLS] → R^1024

    h_fused = torch.cat([h_deb, h_xlmr], dim=-1)  # R^2048
    return h_fused
```

---

## 🏗️ System Architecture

The final **SemanticVectors** system (`final_submission_XLM-Mdeberta-Expert.py`) is a four-stage pipeline:

```
Stage 1 — Data Pooling
    All 22 per-language training CSVs pooled → ~40,395 samples
    (class balance: neutral=1.042, polarized=0.961 — mild imbalance)

Stage 2 — Siamese Dual-Encoder Training (mDeBERTa-v3-large + XLM-RoBERTa-large)
    ┌──────────────────────┐    ┌──────────────────────┐
    │  mDeBERTa-v3-large   │    │  XLM-RoBERTa-large   │
    │  304M params         │    │  560M params         │
    │  LoRA r=32, α=64     │    │  LoRA r=32, α=64     │
    │  4-bit NF4 QLoRA     │    │  4-bit NF4 QLoRA     │
    │  SentencePiece BPE   │    │  XLM BPE vocab       │
    └────────┬─────────────┘    └──────────┬───────────┘
             │ [CLS] ∈ R^1024              │ [CLS] ∈ R^1024
             └──────────── concat ─────────┘
                          h ∈ R^2048
                   LayerNorm → Dropout(0.1)
                   → GELU → Linear → p_hybrid

Stage 3 — Language Expert Models
    GBERT-large       (deu)  — handles ironic/cultural register
    DBMDZ Italian BERT (ita) — handles implicit framing (baseline recall: 0.467)
    AfriBERTa         (swa)  — handles Bantu morphological complexity
    QLoRA r=16, α=32 | Out-of-fold predictions for stacking

Stage 4 — XGBoost Meta-Stacker + Platt Calibration
    Feature vector: [p_hybrid, p_expert, token_count, max(p), H(p)]
                                                              ↑
                                              Shannon entropy — high entropy
                                              → model uncertain → weight expert
    Platt scaling per language → calibrated posteriors
    Per-language threshold t* ∈ [0.1, 0.9] (81-point grid) → ŷ ∈ {0,1}
```

### Loss Function: Focal Loss as a Hard-Example Miner

```python
# Focal Loss with label smoothing (ε=0.05), γ=2.0
# With only 1.08:1 class imbalance, the PRIMARY justification is
# instance difficulty, not label frequency:
#   - γ=2.0 down-weights easily classified examples by up to 96%
#   - Concentrates capacity on implicit rhetorical framing:
#     sarcasm, dog-whistles, ironic rhetorical questions

L = -Σ (1 - p_t)^γ · ỹ · log(p)    where ỹ = label-smoothed target
```

### Dual-Encoder: Why Two Models?

| Property        | mDeBERTa-v3-large                                | XLM-RoBERTa-large                     |
| --------------- | ------------------------------------------------ | ------------------------------------- |
| Pre-training    | ELECTRA-style, gradient-disentangled attention   | Masked LM, 100-language SentencePiece |
| Strength        | Precise syntactic mapping (content vs. position) | Broad cross-lingual semantic coverage |
| Tokenizer       | SentencePiece                                    | BPE                                   |
| Flash Attention | ❌ (use PyTorch SDPA + bfloat16)                 | ✅ Flash Attention 2                  |
| Instruction     | PyTorch SDPA + bfloat16                          | Flash Attention 2 when available      |

The **complementary tokenizers** are a key low-resource benefit: a token unknown to mDeBERTa's SentencePiece is frequently handled by XLM-R's BPE vocabulary, reducing effective OOV rates for morphologically rich scripts (Amharic Ethiopic syllabary, Odia abugida).

---

## 🔄 Development Timeline: Iterative Innovation

The project evolved through **10 distinct phases**. Phases 1–8 informed the final architecture and are reported as negative results / ablations in the paper.

### **Phase 1 — Foundation: BitNet 1.58-bit Baseline** (`Semeval.ipynb`)

> **File:** `Work/Semeval.ipynb`

**Goal:** Establish an efficient quantized English baseline.

```python
class BitLinear(nn.Module):
    """
    1.58-bit Ternary Weight Quantization: weights ∈ {-1, 0, +1}
    - Activations: 8-bit quantization
    - Gradient flow via Straight-Through Estimator (STE)
    - Lambda warmup: λ(t) = min(1.0, t/warmup_steps)
      Mixed = (1 - λ) × FP_weight + λ × Quantized_weight
    - Result: ~10x model compression
    """
    def forward(self, x):
        w_quant = self.quantize_weights()   # ternary {-1,0,1}
        x_quant = self.quantize_activations(x)  # 8-bit
        return F.linear(x_quant, w_quant, self.bias)
```

**Architecture:** `BERT → CLS → BitLinear(768→384) → GELU → Dropout → BitLinear(384→2)`

**Result:** F1 Macro = **0.977** on English. Generalized poorly multilingually due to representational collapse under ternary weights. Established as a **negative result** in the paper.

---

### **Phase 2 — Multilingual Expansion** (`Semeval_multilingual.ipynb`, `Semeval_Optimized.ipynb`)

> **Files:** `Work/Semeval_multilingual.ipynb` · `Work/Semeval_Optimized.ipynb`

**Goal:** Scale BitNet to 9 languages; find optimal classification threshold.

| Notebook                     | Innovation                                       | Result             |
| ---------------------------- | ------------------------------------------------ | ------------------ |
| `Semeval_multilingual.ipynb` | mDeBERTa-v3 backbone, 9-language pooled training | F1 Macro = 0.764   |
| `Semeval_Optimized.ipynb`    | Grid-search threshold t ∈ [0.30, 0.70]           | Optimal t\* = 0.49 |

Key code pattern from threshold optimization:

```python
# Grid search to maximize F1 Macro on validation set
for t in np.arange(0.30, 0.71, 0.01):
    preds = (probs[:, 1] > t).astype(int)
    f1 = f1_score(labels, preds, average="macro")
    if f1 > best_f1:
        best_f1, best_t = f1, t
# → best_t = 0.49
```

---

### **Phase 3 — Advanced Techniques: Data Augmentation + Adapters** (`Semeval_multilingual_IMPROVED.ipynb`)

> **File:** `Work/Semeval_multilingual_IMPROVED.ipynb`

**Goal:** Combat class imbalance and improve low-resource language performance.

```python
# Easy Data Augmentation (EDA) for minority classes
# - Synonym Replacement: 10% of words replaced per sample
# - Target: polarized class in languages where polarized% < 30%
# - Augmentation factor: 2× per minority sample

# Language-specific class weights
for lang in languages:
    n_total = len(df)
    w_neutral   = n_total / (2 * (df.label == 0).sum())
    w_polarized = n_total / (2 * (df.label == 1).sum())

# LoRA language adapters (early version)
LoRAConfig(r=8, lora_alpha=16, target_modules=["query","value"], dropout=0.1)
```

**Outcome:** EDA caused semantic drift in morphologically complex non-English languages. Flagged as a negative result; not used in the final system.

---

### **Phase 4 — Efficiency Breakthrough: RWKV** (`SemEval_RWK.ipynb`)

> **File:** `Work/SemEval_RWK.ipynb`

**Goal:** Replace O(N²) attention with linear-time RWKV for scalability.

```
RWKV Architecture:
  Input → Tokenizer → RWKV Encoder (O(N) Weighted Key-Value) → Pooler → BitLinear Head

Complexity comparison:
  Transformer self-attention:  O(N²)   memory + compute
  RWKV bidirectional WKV:      O(N)    — linear in sequence length

Training results vs. mDeBERTa:
  Time per epoch:  90s  vs.  180s   (2× faster)
  GPU memory:      11.1 GB vs. 15.8 GB (30% reduction)
  F1 Macro:        ~0.75 vs. 0.764  (slightly lower)
```

**Outcome:** 2× faster training, but degraded zero-shot cross-lingual transfer across 22 diverse languages. Reported as a **negative result** in §3.4 of the paper. Representational breadth dominates efficiency for this task.

---

### **Phase 5 — Mamba State-Space Models** (`semeval_mamba.ipynb`)

> **File:** `Work/semeval_mamba.ipynb`

**Goal:** Explore Mamba selective-scan SSM as an alternative to transformers.

The Mamba model uses a selective scan mechanism that allows the hidden state to selectively retain or forget information based on the input, unlike the fixed recurrence in RNNs. Despite the theoretical appeal, it achieved similar efficiency gains as RWKV but identical generalization weaknesses in zero-shot cross-lingual transfer. Reported as a **negative result** alongside RWKV.

---

### **Phase 6 — First Production Pipeline** (`Semantic_mutilingual_deberta.ipynb`)

> **File:** `Work/Semantic_mutilingual_deberta.ipynb`

**Goal:** Build a robust, deployable training infrastructure.

The first end-to-end production pipeline implementing:

```python
# Key training configuration
learning_rate        = 3e-5
batch_size_train     = 16
batch_size_eval      = 64
gradient_accumulation= 2      # effective batch = 32
num_epochs           = 6
warmup_ratio         = 0.06   # 6% linear warmup
max_grad_norm        = 1.0
weight_decay         = 0.02
max_length           = 256    # extended context window
beta2                = 0.98   # AdamW — better multilingual stability

# Infrastructure additions:
# - Stratified 85/15 split (class + language balanced)
# - Mixed-precision AMP (fp16) — 2× memory efficiency
# - Early stopping (patience=3)
# - Per-language F1 evaluation at each checkpoint
# - Best-model checkpointing with automatic restore
```

---

### **Phase 7 — Script Migration & Large Model Evaluation** (`roberta_large.py`, `MT5.py`, `T5Gemma.py`, `MDeberta-XLM.py`, `Mdeberta-QLora.py`)

> **Files:** `Work/roberta_large.py` · `Work/MT5.py` · `Work/T5Gemma.py` · `Work/MDeberta-XLM.py` · `Work/Mdeberta-QLora.py`

**Goal:** Evaluate a broad range of architectures at production scale.

| Script              | Model Type      | Architecture           | Key Finding                                              |
| ------------------- | --------------- | ---------------------- | -------------------------------------------------------- |
| `roberta_large.py`  | Encoder         | RoBERTa Large          | Strong English, weaker low-resource                      |
| `MT5.py`            | Encoder-Decoder | mT5                    | Seq2seq overhead outweighs benefits                      |
| `T5Gemma.py`        | Decoder LLM     | Gemma                  | High VRAM cost, marginal F1 gain                         |
| `MDeberta-XLM.py`   | Dual-encoder    | mDeBERTa + XLM-R       | First dual-model prototype                               |
| `Mdeberta-QLora.py` | Encoder + PEFT  | mDeBERTa + 4-bit QLoRA | Enabled large-model fine-tuning under memory constraints |

`Mdeberta-QLora.py` introduced the 4-bit NF4 double-quantization approach (via `bitsandbytes`) that carried into the final system:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

---

### **Phase 8 — PyTorch Lightning Migration** (`lightning.py`)

> **File:** `Work/lightning.py`

**Goal:** Industrial-grade training framework for multi-GPU scalability.

`lightning.py` refactored the training loop into `LightningModule` and `LightningDataModule` classes, enabling:

- **Distributed Data Parallel (DDP)** training with zero code changes
- Automatic **gradient accumulation, mixed precision, logging, and checkpointing** via Trainer flags
- Clean separation of model logic, data pipeline, and training loop
- Used as the 2nd-to-last submission before expert ensembling was added

---

### **Phase 9 — Final Expert Submission** (`final_submission_XLM-Mdeberta-Expert.py`)

> **File:** `Work/final_submission_XLM-Mdeberta-Expert.py` ← **The definitive submission**

**Goal:** Maximize competition macro-F1 through Siamese encoding + expert ensembling + calibrated meta-stacking.

This script implements the complete four-stage pipeline described in §3 of the paper:

1. Loads both mDeBERTa-v3-large and XLM-RoBERTa-large with 4-bit QLoRA (LoRA r=32, α=64)
2. Trains both encoders jointly on the pooled 22-language corpus with Focal Loss (γ=2.0, ε=0.05)
3. Trains language-specific QLoRA expert models (GBERT, Italian BERT, AfriBERTa)
4. Fits an XGBoost meta-stacker on out-of-fold predictions using `[p_hybrid, p_expert, token_count, max(p), H(p)]`
5. Applies per-language Platt calibration and grid-searched threshold optimization

```python
# Meta-stacker feature construction
import scipy.stats as stats

def build_meta_features(p_hybrid, p_expert, tokens):
    max_p    = np.maximum(p_hybrid, p_expert)
    entropy  = stats.entropy(np.stack([p_hybrid, 1-p_hybrid], axis=1), axis=1)
    return np.stack([p_hybrid, p_expert, tokens, max_p, entropy], axis=1)

# Final threshold optimization per language
for lang in languages:
    best_t = max(
        np.linspace(0.1, 0.9, 81),
        key=lambda t: f1_score(y_dev[lang], (p_cal[lang] > t).astype(int), average="macro")
    )
```

**Result: Macro-F1 = 0.797, Accuracy = 0.827 across 22 languages.**

---

### **Phase 10 — Paper-Ready Code** (`Final_Code_Paper_Submission.py`) _(WIP)_

> **File:** `Work/Final_Code_Paper_Submission.py`

**Goal:** Clean, fully reproducible, modular implementation for academic release.

Finalizing configuration for exact paper reproducibility with unified CLI, deterministic seeding, and full logging.

---

### Version Progression Summary

| Version Stage   | File                                      | Key Method                 | Performance                  |
| --------------- | ----------------------------------------- | -------------------------- | ---------------------------- |
| **Initial v1**  | `Semeval.ipynb`                           | BitNet 1.58-bit on English | F1: 0.977 (EN)               |
| **Multi v1**    | `Semeval_multilingual.ipynb`              | BitNet 9-language          | F1: 0.764 (Multi)            |
| **Multi v1.1**  | `Semeval_Optimized.ipynb`                 | Threshold grid-search      | t\* = 0.49                   |
| **Improved v2** | `Semeval_multilingual_IMPROVED.ipynb`     | EDA + LoRA adapters        | Enhanced (EDA later dropped) |
| **RWKV**        | `SemEval_RWK.ipynb`                       | O(N) architecture          | 2× faster, neg. result       |
| **Mamba**       | `semeval_mamba.ipynb`                     | Selective-scan SSM         | Experimental, neg. result    |
| **Stable v1**   | `Semantic_mutilingual_deberta.ipynb`      | Production XLM-R pipeline  | Production-ready             |
| **Scaling**     | `Mdeberta-QLora.py` / `MT5.py`            | 4-bit QLoRA, generative    | Advanced tuning              |
| **2nd Last**    | `lightning.py`                            | PyTorch Lightning DDP      | 2nd-last version             |
| **Final**       | `final_submission_XLM-Mdeberta-Expert.py` | Siamese + XGBoost stacker  | **F1: 0.797 (22 lang)**      |
| **Paper WIP**   | `Final_Code_Paper_Submission.py`          | Reproducible release       | WIP                          |

---

## 📊 Results & Performance

### Official Test Set Results — SV-FULL (22 Languages)

| Language         | Accuracy | Precision | Recall | F1 Macro |
| ---------------- | -------- | --------- | ------ | -------- |
| Amharic (amh)    | .842     | .866      | .930   | .780     |
| Arabic (arb)     | .831     | .789      | .850   | .830     |
| Bengali (ben)    | .831     | .797      | .804   | .827     |
| German (deu)     | .726     | .697      | .757   | .726     |
| English (eng)    | .809     | .721      | .784   | .798     |
| Persian (fas)    | .851     | .893      | .907   | .803     |
| Hausa (hau)      | .924     | .639      | .657   | .803     |
| Hindi (hin)      | .901     | .948      | .935   | .811     |
| Italian (ita)    | .661     | .717      | .467   | .644     |
| Khmer (khm)      | .922     | .952      | .962   | .755     |
| Myanmar (mya)    | .864     | .861      | .909   | .860     |
| **Nepali (nep)** | **.909** | .924      | .891   | **.909** |
| Odia (ori)       | .817     | .690      | .647   | .771     |
| Punjabi (pan)    | .768     | .756      | .771   | .768     |
| Polish (pol)     | .806     | .776      | .754   | .800     |
| Russian (rus)    | .805     | .659      | .718   | .773     |
| Spanish (spa)    | .782     | .766      | .803   | .782     |
| Swahili (swa)    | .780     | .817      | .725   | .780     |
| Telugu (tel)     | .873     | .888      | .864   | .873     |
| Turkish (tur)    | .789     | .805      | .784   | .789     |
| Urdu (urd)       | .807     | .856      | .868   | .771     |
| Chinese (zho)    | .886     | .892      | .882   | .886     |
| **System Avg**   | **.827** | —         | —      | **.797** |

**Strongest languages:** Nepali (.909), Chinese (.886), Telugu (.873) — morphologically consistent, script-uniform.  
**Weakest languages:** Italian (.644), German (.726) — implicit ironic framing.

### Ablation Study

| Configuration              | F1 Macro  | Δ    |
| -------------------------- | --------- | ---- |
| mDeBERTa-v3-large alone    | 0.762     | —    |
| XLM-R-large alone          | 0.771     | +0.9 |
| Siamese dual-encoder       | 0.789     | +1.8 |
| + threshold optimization   | 0.795     | +0.6 |
| + expert ensembling (FULL) | **0.797** | +0.2 |
| FULL − focal loss (CE)     | 0.781     | −1.6 |
| FULL − label smoothing     | 0.785     | −1.2 |

The Siamese dual-encoder delivers the largest single gain (+1.8 pp). Focal loss is the second most critical component: ablating it costs −1.6 pp.

### Task Baseline Comparison

| System                                | Macro-F1  |
| ------------------------------------- | --------- |
| Official task majority-class baseline | 0.461     |
| Single mDeBERTa-v3-large              | 0.762     |
| Single XLM-R-large                    | 0.771     |
| **SemanticVectors (FULL)**            | **0.797** |

---

## 🔬 Technical Details

### Final Hyperparameters (`final_submission_XLM-Mdeberta-Expert.py`)

```python
# Dual-encoder backbone settings
backbones          = ["microsoft/mdeberta-v3-base", "xlm-roberta-large"]
quantization       = "4-bit NF4 (double quantization)"
compute_dtype      = torch.bfloat16        # H100 native; more stable than float16
lora_r_hybrid      = 32
lora_alpha_hybrid  = 64
lora_r_expert      = 16
lora_alpha_expert  = 32
target_modules     = ["query_proj", "value_proj"]  # mDeBERTa
                   # ["query", "value"]             # XLM-R

# Training
num_epochs         = 4
batch_size         = 64
lr_hybrid          = 2e-4
lr_expert          = 2e-5
max_seq_length     = 256
warmup_ratio       = 0.25   # 25% cosine annealing warm-up
optimizer          = "AdamW"
scheduler          = "CosineAnnealing"

# Loss
focal_gamma        = 2.0
label_smoothing    = 0.05

# Meta-stacker
meta_model         = "XGBoost + Platt calibration (logistic regression)"
threshold_grid     = np.linspace(0.1, 0.9, 81)   # per language

# Hardware
gpu                = "NVIDIA H100 (80 GB)"
inference_vram     = "~28 GB  (14 GB per 4-bit model)"
throughput         = "~340 samples/sec  @ batch=32, seq=256"
```

### Key Technical Patterns

#### 4-bit QLoRA Loading

```python
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,   # double quantization saves ~0.4 bits/param
    bnb_4bit_quant_type="nf4",        # NF4 better than FP4 for transformer weights
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModel.from_pretrained("xlm-roberta-large", quantization_config=bnb_config)

lora_cfg = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none", task_type="SEQ_CLS"
)
model = get_peft_model(model, lora_cfg)
```

#### Focal Loss with Label Smoothing

```python
class FocalLossWithSmoothing(nn.Module):
    def __init__(self, gamma=2.0, eps=0.05):
        super().__init__()
        self.gamma, self.eps = gamma, eps

    def forward(self, logits, targets):
        n_cls = logits.size(-1)
        # Apply label smoothing
        smooth_targets = torch.full_like(logits, self.eps / (n_cls - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.eps)

        log_p  = F.log_softmax(logits, dim=-1)
        p_t    = torch.exp(log_p).gather(1, targets.unsqueeze(1))
        weight = (1 - p_t) ** self.gamma       # hard-example focusing

        loss = -(weight * (smooth_targets * log_p).sum(-1, keepdim=True))
        return loss.mean()
```

#### XGBoost Meta-Stacker Feature Construction

```python
import numpy as np
from scipy.stats import entropy
import xgboost as xgb

def build_meta_features(p_hybrid, p_expert, token_lengths):
    """
    Five-dimensional feature vector per sample:
      p_hybrid      — Siamese dual-encoder soft probability (polarized class)
      p_expert      — Language-specific expert soft probability
      token_lengths — Subword token count (proxy for text complexity)
      max_conf      — max(p_hybrid, p_expert) — prediction confidence
      H(p)          — Shannon entropy over [p, 1-p] of hybrid prediction
                       High entropy → model uncertain → stacker weights expert more
    """
    max_conf = np.maximum(p_hybrid, p_expert)
    H = entropy(np.stack([p_hybrid, 1 - p_hybrid], axis=1).T)
    return np.stack([p_hybrid, p_expert, token_lengths, max_conf, H], axis=1)

meta_X = build_meta_features(p_hyb_oof, p_exp_oof, tok_counts)
meta_clf = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05)
meta_clf.fit(meta_X, y_train)
```

---

## 🌍 Dataset

### POLAR@SemEval-2026 Task 9 — Subtask 1

**Task:** Binary classification — Polarized (1) vs. Neutral (0)

**Labels:**

- `0` — Not Polarized: neutral, analytical, or informational content
- `1` — Polarized: divisive framing, out-group vilification, in-group solidarity, dehumanizing language

**Format:**

```csv
id,text,label
1,"This is a neutral news report about the election.",0
2,"These people are destroying everything we built!",1
```

**Languages — 22 across 8 typological families:**

| Family                    | Languages                    |
| ------------------------- | ---------------------------- |
| Western European          | deu, eng, ita, spa, pol      |
| Indic / Indo-Aryan        | hin, nep, ori, pan, tel, urd |
| Semitic                   | arb, fas                     |
| African                   | amh, hau, swa                |
| Southeast Asian           | khm, mya                     |
| Sinitic / Turkic / Slavic | zho / tur / rus, ben         |

**Statistics:**

- Training: ~40,395 pooled samples across 22 languages
- Class balance: neutral=1.042, polarized=0.961 (mild, 1.08:1 ratio)
- Low-resource threshold: < 500 samples (hau, ori, amh, swa) — systematic over-prediction bias observed

**Data files are organized as:**

```
Work/subtask1/train/{lang}.csv   # full 22-language competition data
Work/subtask1/dev/{lang}.csv
dev_phase_data/subtask1/train/{lang}.csv  # early 9-language dev-phase data
```

---

## 🔍 Key Innovations

### 1. Siamese Dual-Encoder Architecture

Two large encoders process the same input **independently in parallel**, then their `[CLS]` representations are concatenated to form a dense $\mathbb{R}^{2048}$ joint embedding:

```
h = [h_mDeBERTa ; h_XLM-R] ∈ R^{2048}
Classification head: LayerNorm → Dropout(0.1) → GELU → Linear(2048, 2)
```

**Why this works:**

- mDeBERTa's **disentangled attention** (content vs. position) provides precise syntactic mapping
- XLM-R's **broad 100-language pretraining** provides cross-lingual semantic coverage
- The 2048-dim fusion is robust to ultra-short texts (macro-F1 ~0.82 on 0–15 word sequences), disproving the hypothesis that short texts create a representational bottleneck
- **Complementary tokenizers** (SentencePiece + BPE) jointly cover rare or morphologically complex sub-words, reducing OOV for Amharic, Odia, and other low-resource scripts

### 2. XGBoost Meta-Stacker with Platt Calibration

Rather than simple soft-vote averaging, the meta-stacker ingests a five-dimensional feature vector per sample. The Shannon entropy signal is the key design feature:

```
H(p) = -p·log(p) - (1-p)·log(1-p)

High entropy → Siamese encoder is uncertain → stacker increases expert weight
Low entropy  → Siamese encoder is confident  → stacker trusts hybrid prediction
```

After stacking, a per-language **Platt scaling** logistic regression calibrates raw XGBoost probabilities to match the development-set class distribution, substantially mitigating over-prediction bias in low-resource languages.

### 3. Focal Loss as a Hard-Example Miner

With only a 1.08:1 class imbalance ratio, focal loss is **not** used to correct label frequency — it functions as a hard-example miner:

```
γ=2.0:  easy examples (p_t ≈ 0.9) are down-weighted by (1-0.9)^2 = 0.01 → 99% weight reduction
        hard examples (p_t ≈ 0.5) carry full gradient signal

Effect: model capacity is concentrated on subtly framed rhetorical content —
        sarcasm, cultural dog-whistles, ironic rhetorical questions —
        rather than lexically obvious polarization
```

### 4. Per-Language Threshold Optimization

Languages have widely different training set sizes (thousands for English vs. < 500 for Hausa) and class priors. A universal threshold of 0.5 is suboptimal:

```python
# Grid search 81 thresholds per language
for lang in languages:
    if len(y_dev[lang]) < 10:
        t_opt[lang] = 0.5  # insufficient dev data → default
        continue
    t_opt[lang] = max(
        np.linspace(0.1, 0.9, 81),
        key=lambda t: f1_score(y_dev[lang],
                               (p_cal[lang] > t).astype(int),
                               average="macro")
    )
```

---

## 📈 Inference & Deployment

### Computational Cost

| Configuration                 | VRAM          | Throughput         | F1 Macro              |
| ----------------------------- | ------------- | ------------------ | --------------------- |
| SV-FULL (Siamese + experts)   | ~28 GB (H100) | ~340 samples/sec   | **0.797**             |
| SV-BASE (Siamese, no experts) | ~14 GB        | ~680 samples/sec   | ~0.795 (~96% of full) |
| mDeBERTa alone                | ~7 GB         | ~1,200 samples/sec | 0.762                 |

> For latency-sensitive deployments, **SV-BASE** retains ~96% of full-system performance at half the VRAM cost.

### Training Time (Full Pipeline on H100)

| Stage                                      | Time         |
| ------------------------------------------ | ------------ |
| Dual-encoder training (4 epochs, batch=64) | ~2.5 hours   |
| Expert model training (3 × 4 epochs)       | ~1.5 hours   |
| XGBoost stacker + Platt calibration        | < 5 minutes  |
| Threshold optimization (per-language)      | < 2 minutes  |
| **Total**                                  | **~4 hours** |

---

## 📦 Dependencies

```
torch>=2.0.0
transformers>=4.40.0
peft                    # LoRA / QLoRA adapters
bitsandbytes            # 4-bit NF4 quantization
accelerate              # Mixed-precision + distributed training
xgboost                 # Meta-stacker
scikit-learn            # Platt calibration + metrics
scipy                   # Shannon entropy computation
pandas
numpy
```

Install all:

```bash
pip install torch transformers peft bitsandbytes accelerate xgboost scikit-learn scipy pandas numpy
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- POLAR@SemEval-2026 Task 9 organizers for datasets and evaluation infrastructure
- Google Colab, Kaggle, and Lightning AI for GPU resources during early development
- Anonymous reviewers for constructive feedback on the paper
