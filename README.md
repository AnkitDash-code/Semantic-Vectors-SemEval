# SemEval 2025 Task 11: Multilingual Polarization Detection with BitNet

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.40.0-orange.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

This repository contains our solution for **SemEval 2025 Task 11: Subtask 1 - Binary Polarization Detection**. We implement a novel approach combining **BitNet 1.58-bit quantization** with state-of-the-art transformer architectures for efficient multilingual text polarization classification.

📄 **Research Abstract**: A detailed overview of our methodology, architecture, and results is available in [`GenAI_Semeval_Abstract.pdf`](GenAI_Semeval_Abstract.pdf).

### Key Achievements

- 🏆 **F1 Macro: 0.977** on English validation set
- 🌍 **9 Languages Supported**: English, Arabic, German, Spanish, Italian, Urdu, Chinese, Hausa, Amharic
- ⚡ **2x Faster Training** with RWKV architecture variant
- 💾 **30% Memory Reduction** compared to standard transformers
- 🎯 **Multilingual F1 Macro: 0.764** across all languages

---

## 📁 Project Structure

```
Work/
│
├── Notebooks/ (Initial Exploration & Prototypes)
│   ├── Semeval.ipynb                          # Foundation: BERT + BitNet (English only)
│   ├── Semeval_multilingual.ipynb             # Multilingual extension
│   ├── Semeval_Optimized.ipynb                # Hyperparameter optimization
│   ├── Semeval_multilingual_IMPROVED.ipynb    # Advanced features + data augmentation
│   ├── SemEval_RWK.ipynb                      # RWKV efficient architecture
│   ├── semeval_mamba.ipynb                    # Mamba state-space model (experimental)
│   ├── Semantic_mutilingual_deberta.ipynb     # Early Production XLM-RoBERTa pipeline
│   └── Roberta_model2.ipynb                   # RoBERTa experiments
│
├── Scripts/ (Model Scaling & Advanced Tuning)
│   ├── new_model.py & new_model_fixed.py      # Baseline script migration
│   ├── roberta_large.py                       # RoBERTa Large baseline
│   ├── MT5.py & T5Gemma.py                    # Sequence-to-Sequence / Generative approaches
│   ├── MDeberta-XLM.py & Mdeberta-XLM-final.py# XLM-RoBERTa & mDeBERTa integration
│   └── Mdeberta-QLora.py                      # QLoRA efficient fine-tuning
│
├── Final Submissions/ (Current & Future)
│   ├── lightning.py                           # 2nd Last Version: PyTorch Lightning framework migration
│   ├── final_submission_XLM-Mdeberta-Expert.py# Last & Final Version: Ensembled XLM/mDeBERTa
│   └── [WIP] paper_submission_version.py      # Final Paper Version: Reproducible pipeline (Code WIP)
│
├── Papers/                                    # Relevant research papers
├── GenAI_Semeval_Abstract.pdf                 # Research abstract and methodology overview
├── project_timeline.csv                       # Development phases and timeline
│
├── subtask1/                                  # Dataset directory
│   ├── train/                                 # Training data (9 language CSVs)
│   └── dev/                                   # Development data
│
├── predictions_qwen3/                         # Inference outputs from external models
└── README.md                                  # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AnkitDash-code/Semantic-Vectors-SemEval
cd semeval-polarization

# Install dependencies
pip install transformers==4.40.0 torch==2.0.0 accelerate scikit-learn pandas numpy
```

### Basic Usage

```python
from transformers import AutoTokenizer
import torch

# Load trained model
model = BitNetBinaryClassifier(
    model_name="microsoft/mdeberta-v3-base",
    num_labels=2,
    dropout_prob=0.2
)
model.load_state_dict(torch.load("path/to/model.bin"))
tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")

# Predict
text = "This politician is destroying our country!"
pred, confidence = predict_polarization(text, model, tokenizer)
print(f"Prediction: {'Polarized' if pred == 1 else 'Not Polarized'}")
print(f"Confidence: {confidence:.3f}")
```

---

## 🏗️ Architecture Evolution

### Development Timeline: Iterative Innovation

The project evolved through **10 distinct phases**, each building upon the previous to create a comprehensive multilingual polarization detection system:

#### **Phase 1: Foundation (Initial Version)**

🎯 **Goal**: Establish baseline with efficient quantization

| Notebook        | Innovation                   | Architecture     | Performance             |
| --------------- | ---------------------------- | ---------------- | ----------------------- |
| `Semeval.ipynb` | BitNet 1.58-bit quantization | BERT + BitLinear | **F1: 0.977** (English) |

**Key Achievements**:

- Introduced 1.58-bit ternary quantization {-1, 0, 1}
- Implemented Focal Loss for class imbalance
- Developed Lambda warmup for gradual quantization
- Achieved 97.7% F1 Macro on English validation set

---

#### **Phase 2: Multilingual Expansion (Early Versions)**

🌍 **Goal**: Scale to 9 languages with optimized hyperparameters

| Notebook                     | Innovation             | Architecture            | Performance                  |
| ---------------------------- | ---------------------- | ----------------------- | ---------------------------- |
| `Semeval_multilingual.ipynb` | 9-language support     | mDeBERTa-v3 + BitLinear | **F1: 0.764** (Multilingual) |
| `Semeval_Optimized.ipynb`    | Threshold optimization | Same as above           | Optimal threshold: **0.49**  |

**Key Achievements**:

- Extended to 9 languages: English, Arabic, German, Spanish, Italian, Urdu, Chinese, Hausa, Amharic
- Processed 29,987 training samples
- Implemented language-agnostic prediction pipeline
- Fine-tuned classification threshold for F1 maximization

---

#### **Phase 3: Advanced Techniques (Intermediate Versions)**

🚀 **Goal**: Enhance model with data augmentation and adaptive features

| Notebook                              | Innovation                   | Architecture                     | Performance                                 |
| ------------------------------------- | ---------------------------- | -------------------------------- | ------------------------------------------- |
| `Semeval_multilingual_IMPROVED.ipynb` | Data aug + Language adapters | mDeBERTa-v3 + Enhanced BitLinear | **Enhanced F1** + Better imbalance handling |

**Key Achievements**:

- Implemented Easy Data Augmentation (EDA) for minority classes
- Added language-specific class weights
- Integrated LoRA adapters for fine-grained language tuning
- Extended context window to 192 tokens
- Employed cosine learning rate scheduling with warmup

---

#### **Phase 4: Efficiency Breakthrough (Experimental)**

⚡ **Goal**: Achieve linear complexity with competitive accuracy

| Notebook            | Innovation             | Architecture     | Performance                  |
| ------------------- | ---------------------- | ---------------- | ---------------------------- |
| `SemEval_RWK.ipynb` | RWKV O(N) architecture | RWKV + BitLinear | **2x faster** + 30% memory ↓ |

**Key Achievements**:

- Integrated RWKV with bidirectional Weighted Key-Value attention
- Reduced complexity from O(N²) to **O(N)**
- Achieved 2x training speed (90s vs 180s per epoch)
- Cut GPU memory usage by 30% (11.1 GB vs 15.8 GB)
- Maintained competitive F1 Macro (~0.75)

---

#### **Phase 5: Experimental Exploration (Alternative Models)**

🔬 **Goal**: Explore next-generation architectures

| Notebook              | Innovation              | Architecture      | Status           |
| --------------------- | ----------------------- | ----------------- | ---------------- |
| `semeval_mamba.ipynb` | Mamba state-space model | Mamba + BitLinear | **Experimental** |

**Key Achievements**:

- Explored selective scan mechanism for sequence modeling
- Investigated state-space models as transformer alternatives
- Laid groundwork for future architecture research
- Identified potential for hybrid approaches

---

#### **Phase 6: Production-Ready Pipeline (First Stable Version)**

🏭 **Goal**: Create robust, deployable training infrastructure

| Notebook                             | Innovation                   | Architecture                         | Performance          |
| ------------------------------------ | ---------------------------- | ------------------------------------ | -------------------- |
| `Semantic_mutilingual_deberta.ipynb` | Production training pipeline | XLM-RoBERTa-base + Advanced Training | **Production-Ready** |

**Key Achievements**:

- Implemented **XLM-RoBERTa-base** as multilingual backbone (better than mDeBERTa for deployment)
- **Stratified split** (85/15) maintaining class and language balance
- **Balanced class weights** computed per-dataset for optimal F1
- **Mixed precision training (AMP)** for 2x memory efficiency
- **Gradient accumulation (2x)** enabling larger effective batch sizes
- **Early stopping** with patience=3 to prevent overfitting
- **Linear warmup (6%)** + decay scheduling for stable convergence
- **Per-language F1 evaluation** for fine-grained performance tracking
- **Best model checkpointing** with automatic save/restore
- Extended context to **256 tokens** for longer documents
- AdamW optimizer with **β2=0.98** for better multilingual stability

**Training Configuration**:

```python
learning_rate = 3e-5
per_device_train_batch_size = 16
per_device_eval_batch_size = 64
gradient_accumulation_steps = 2
num_epochs = 6
warmup_ratio = 0.06
max_grad_norm = 1.0
weight_decay = 0.02
```

**Production Features**:

- Robust error handling and data validation
- Efficient DataLoader with pin_memory optimization
- Comprehensive evaluation metrics (overall + per-language)
- Model versioning and checkpoint management
- Ready for deployment pipeline integration

---

#### **Phase 7: Script Migration & Advanced Architectures (Scaling)**

🚀 **Goal**: Transition from notebooks to robust scripts and explore large/generative models

| Script                              | Innovation                      | Architecture     | Status       |
| ----------------------------------- | ------------------------------- | ---------------- | ------------ |
| `roberta_large.py` / `new_model.py` | Scaling up parameters           | RoBERTa Large    | Evaluated    |
| `MT5.py` / `T5Gemma.py`             | Seq2seq & Generative approaches | mT5 / Gemma      | Evaluated    |
| `MDeberta-XLM.py`                   | Multilingual model integration  | mDeBERTa + XLM-R | Pre-final    |
| `Mdeberta-QLora.py`                 | Parameter-efficient fine-tuning | mDeBERTa + QLoRA | Optimization |

**Key Achievements**:

- Migrated experimental notebooks to scalable Python scripts.
- Evaluated decoder/encoder-decoder architectures (Gemma, mT5) for classification.
- Leveraged QLoRA for memory-efficient training of large checkpoint models.

---

#### **Phase 8: PyTorch Lightning Migration (2nd Last Version)**

⚡ **Goal**: Industrial-grade training framework scalability

| Script         | Innovation                    | Architecture     | Status               |
| -------------- | ----------------------------- | ---------------- | -------------------- |
| `lightning.py` | PyTorch Lightning integration | Modular Pipeline | **2nd Last Version** |

**Key Achievements**:

- Re-architected training loop into PyTorch Lightning modules for multi-GPU efficiency.
- Streamlined distributed data parallel (DDP) training.
- Handled advanced gradient accumulation, logging, and callbacks automatically.

---

#### **Phase 9: Final Expert Submission (Final Version)**

🏆 **Goal**: Maximizing competition metrics via specialized routing/ensembling

| Script                                    | Innovation               | Architecture           | Status            |
| ----------------------------------------- | ------------------------ | ---------------------- | ----------------- |
| `final_submission_XLM-Mdeberta-Expert.py` | Hybrid Expert Ensembling | XLM-RoBERTa + mDeBERTa | **Final Version** |

**Key Achievements**:

- Created the definitive submission script utilizing a Mixture of Experts or strict ensembling between XLM-R and mDeBERTa models.
- Yielded the highest overall multilingual F1 Macro score.

---

#### **Phase 10: Official Paper Release (WIP)**

📝 **Goal**: Prepare clean, modular code for academic publishing

| Script                        | Innovation                    | Target                | Status               |
| ----------------------------- | ----------------------------- | --------------------- | -------------------- |
| `paper_submission_version.py` | Cleaned reproducible pipeline | Refined Architectures | **Work In Progress** |

**Key Achievements**:

- Finalizing configuration for exact paper reproducibility.
- Code is currently actively being worked on and will be pushed later.

---

### Version Progression Summary

| Version Stage    | File / Script                         | Key Innovation                | Performance          |
| ---------------- | ------------------------------------- | ----------------------------- | -------------------- |
| **Initial v1**   | `Semeval.ipynb`                       | Initial BitNet implementation | F1: 0.977 (EN)       |
| **Multi v1**     | `Semeval_multilingual.ipynb`          | 9-language support            | F1: 0.764 (Multi)    |
| **Multi v1.1**   | `Semeval_Optimized.ipynb`             | Hyperparameter tuning         | Threshold: 0.49      |
| **Improved v2**  | `Semeval_multilingual_IMPROVED.ipynb` | Data aug + adapters           | Enhanced             |
| **RWKV Branch**  | `SemEval_RWK.ipynb`                   | RWKV O(N) architecture        | 2x faster            |
| **Mamba Branch** | `semeval_mamba.ipynb`                 | Mamba SSM exploration         | Experimental         |
| **Stable v1**    | `Semantic_mutilingual_deberta.ipynb`  | Production pipeline           | Production-Ready     |
| **Scaling v1**   | `Mdeberta-QLora.py` / `MT5.py`        | Scripts & advanced models     | Advanced Tuning      |
| **2nd Last v1**  | `lightning.py`                        | PyTorch Lightning migration   | **2nd Last Version** |
| **Final v1**     | `final_submission_..._Expert.py`      | Hybrid Expert Ensembling      | **Final Version**    |
| **Paper WIP**    | `paper_submission_version.py`         | Reproducible Paper Release    | **WIP**              |

### Core Components

#### 1. **BitLinear Quantization**

```python
class BitLinear(nn.Module):
    """
    1.58-bit Quantized Linear Layer
    - Weights: Ternary {-1, 0, 1}
    - Activations: 8-bit quantization
    - Straight-Through Estimator (STE) for gradient flow
    - Lambda warmup for gradual quantization
    """
```

**Benefits:**

- Reduces model size by ~10x
- Maintains competitive accuracy
- Faster inference on specialized hardware

#### 2. **Model Architectures**

##### Standard BitNet (BERT/mDeBERTa)

```
Input Text → Tokenizer → BERT/mDeBERTa Encoder → CLS Token
    → BitLinear Layer 1 (768 → 384) → GELU → Dropout
    → BitLinear Layer 2 (384 → 2) → Softmax → Prediction
```

##### RWKV Variant (Efficient)

```
Input Text → Tokenizer → RWKV Encoder (O(N) complexity) → Pooler
    → BitLinear Head → Prediction
```

**RWKV Advantages:**

- Linear time complexity O(N) vs O(N²) for transformers
- 2x faster training per epoch
- 30% less GPU memory usage
- Scales to 2048+ token sequences

---

## 📊 Results & Performance

### English Validation Results

| Model Variant    | F1 Macro  | F1 Binary | Accuracy | Threshold |
| ---------------- | --------- | --------- | -------- | --------- |
| BitNet-BERT      | **0.977** | 0.972     | 0.978    | 0.50      |
| BitNet-Optimized | 0.975     | 0.970     | 0.976    | 0.49      |

### Multilingual Validation Results

| Language      | Samples    | Polarized % | F1 Macro  |
| ------------- | ---------- | ----------- | --------- |
| English (eng) | 2,676      | 37.4%       | 0.821     |
| Arabic (arb)  | 3,380      | 44.7%       | 0.756     |
| German (deu)  | 3,180      | 47.5%       | 0.743     |
| Spanish (spa) | 3,305      | 50.2%       | 0.768     |
| Italian (ita) | 3,334      | 41.0%       | 0.761     |
| Urdu (urd)    | 2,849      | 69.4%       | 0.724     |
| Chinese (zho) | 4,280      | 49.6%       | 0.752     |
| Hausa (hau)   | 3,651      | 10.7%       | 0.688     |
| Amharic (amh) | 3,332      | 75.6%       | 0.701     |
| **Overall**   | **29,987** | **46.9%**   | **0.764** |

---

## 🔬 Technical Details

### Training Configuration

#### Optimized Hyperparameters

```python
TrainingArguments(
    num_train_epochs=6,
    per_device_train_batch_size=32,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    fp16=True,  # Mixed precision
    gradient_accumulation_steps=1,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro"
)
```

#### Loss Functions

**Focal Loss** (Default)

```python
FocalLoss(alpha=0.65, gamma=2.0)
```

- Handles class imbalance effectively
- Focuses on hard-to-classify examples
- Better than weighted Cross-Entropy for F1 Macro optimization

**Language-Aware Focal Loss** (IMPROVED variant)

- Language-specific alpha/gamma parameters
- Adapts to varying class distributions per language

### Data Augmentation (IMPROVED Variant)

```python
# Easy Data Augmentation (EDA)
- Synonym Replacement: 10% of words
- Target: Minority class in low-resource languages
- Augmentation Factor: 2x per minority sample
```

### Advanced Features

#### 1. **Lambda Warmup Schedule**

Gradual quantization from full precision to 1.58-bit:

```
λ(t) = min(1.0, t / warmup_steps)
Mixed = (1 - λ) × Full_Precision + λ × Quantized
```

#### 2. **Optimal Threshold Searching**

```python
# Grid search from 0.30 to 0.70
threshold_range = np.arange(0.30, 0.71, 0.01)
best_threshold = 0.49  # Maximizes F1 Macro
```

#### 3. **Language-Specific Class Weights**

```python
# Compute per-language inverse frequency weights
weight[class] = total_samples / (2 × class_count)
# Normalized across classes
```

#### 4. **LoRA Adapters** (Optional)

```python
# Low-Rank Adaptation for language-specific tuning
LoRAConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1
)
```

---

## 📈 Training Timeline & Complexity

### Training Time Comparison

| Model Variant | Samples | Epochs | Time/Epoch | Total Time | GPU Memory |
| ------------- | ------- | ------ | ---------- | ---------- | ---------- |
| BitNet-BERT   | 2,676   | 5      | 58s        | 290s       | 15.8 GB    |
| BitNet-Multi  | 29,987  | 3      | 180s       | 540s       | 15.8 GB    |
| RWKV-Multi    | 29,987  | 3      | 90s        | 270s       | 11.1 GB    |

### Computational Complexity

| Component      | Standard Transformer | RWKV | Improvement |
| -------------- | -------------------- | ---- | ----------- |
| Self-Attention | O(N²)                | O(N) | **Linear**  |
| Memory         | O(N²)                | O(N) | **~70%**    |
| Inference      | O(N²)                | O(N) | **~50%**    |

---

## 🛠️ Advanced Usage

### Training from Scratch

```python
# Run multilingual training
model, tokenizer, trainer, results = train_multilingual_polarization_detector(
    train_dir="data/subtask1/train/",
    languages=None,  # All 9 languages
    model_name="microsoft/mdeberta-v3-base",
    use_data_augmentation=True,
    use_language_specific_weights=True,
    max_length=192,
    num_epochs=6,
    learning_rate=5e-5
)
```

### Generate Predictions

```python
# Generate multilingual predictions
predictions = generate_multilingual_predictions(
    model=model,
    tokenizer=tokenizer,
    dev_dir="data/subtask1/dev/",
    output_dir="predictions/",
    languages=None,
    threshold=0.49
)
```

### Threshold Optimization

```python
# Find optimal threshold for validation set
optimal_threshold, best_f1, results_df = find_optimal_threshold(
    model=model,
    tokenizer=tokenizer,
    val_file="data/subtask1/train/eng.csv"
)
print(f"Optimal Threshold: {optimal_threshold:.2f}")
print(f"Expected F1 Macro: {best_f1:.4f}")
```

---

## 🧪 Experimental Variants

### RWKV Architecture

- **File**: `SemEval_RWK.ipynb`
- **Innovation**: Bidirectional WKV (Weighted Key-Value) attention
- **Complexity**: O(N) instead of O(N²)
- **Trade-off**: Slightly lower F1 (~0.75) but 2x faster

### Mamba State-Space Model

- **File**: `semeval_mamba.ipynb`
- **Innovation**: Selective scan mechanism
- **Status**: Experimental, research phase
- **Goal**: Explore alternatives to transformer architectures

---

## 📚 Dataset

### SemEval 2025 Task 9 - Subtask 1

**Task**: Binary classification of text polarization

**Labels**:

- `0`: Not Polarized
- `1`: Polarized (divisive, inflammatory, or biased content)

**Format**:

```csv
id,text,polarization
1,"This is a neutral statement.",0
2,"Those people are destroying everything!",1
```

**Languages**: 9 (eng, arb, deu, spa, ita, urd, zho, hau, amh)

**Statistics**:

- Training: 29,987 samples
- Class Distribution: 46.9% polarized, 53.1% non-polarized
- Imbalance varies by language (Hausa: 10.7%, Amharic: 75.6%)

---

## 🔍 Key Innovations

### 1. Siamese Dual-Encoder Architecture

- **mDeBERTa-v3-large + XLM-RoBERTa-large**: Jointly fine-tuned via 4-bit QLoRA.
- **Complementary Tokenization**: Fusing SentencePiece and BPE vocabularies reduces out-of-vocabulary (OOV) rates, particularly improving representations for morphologically rich or low-resource scripts like Amharic and Odia.
- **Joint Representation**: The `[CLS]` vectors form a dense $\mathbb{R}^{2048}$ fusion, yielding high robustness even on ultra-short texts ($0\text{-}15$ words), neutralizing the sequence length bottleneck.

### 2. XGBoost Meta-Stacker with Platt Calibration

- **Feature-Rich Fusion**: The meta-stacker moves beyond soft voting by ingesting prediction confidences ($p_{hyb}$, $p_{exp}$), token counts, and **Shannon entropy**.
- **Shannon Entropy as Reliability**: High-entropy predictions signal model uncertainty, triggering the stacker to dynamically weight language-specific experts (e.g., GBERT, Italian BERT).
- **Two-Stage Calibration**: Applies per-language Platt Scaling to calibrate posterior probabilities before thresholding, substantially mitigating over-prediction bias in low-resource setups.

### 3. Focal Loss as a Hard-Example Miner

- Unlike addressing standard class imbalance, our implementation of Focal Loss (with Label Smoothing) acts as a **hard-example miner**.
- It down-weights easily identifiable explicit toxicity, heavily concentrating model gradients on the high-entropy, subtly framed sentences (e.g., sarcasm, irony, cultural dog-whistles) responsible for cross-lingual polarization.

### 4. Per-Language Threshold Optimization

- To account for language-specific skews and data scarcity profiles (which range from massive English datasets to fewer than 500 samples in Hausa and Odia).
- Employs a grid-searched optimal threshold tuned distinctively over the development sets.

---

## 📦 Dependencies

```
torch>=2.0.0
transformers>=4.40.0
peft
accelerate
xgboost
scikit-learn
pandas
numpy
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- SemEval-2026 Task 9 organizers
