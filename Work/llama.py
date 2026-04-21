import os
import glob
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
#from google.colab import drive

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================
# Mount Google Drive
#drive.mount('/content/drive')

# Authenticate with Hugging Face (Replace with your actual token)
login(token="insert your token here")

# Define your Google Drive paths (Update these if your folder name is different)
BASE_DIR = "/content/drive/MyDrive/SemEval_Task9_Data" 
DEV_DIR = f"{BASE_DIR}/dev"

# ==========================================
# 2. LOAD LLAMA-3 IN 4-BIT (To fit on Colab)
# ==========================================
print("Loading Llama-3-8B-Instruct in 4-bit...")
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# 4-bit Quantization makes the 8B model fit perfectly on a 16GB T4 GPU
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()

# ==========================================
# 3. DEFINING THE ZERO-SHOT PROMPT
# ==========================================
def build_prompt(text):
    """
    We use Llama-3's native chat template for best instruction following.
    We strictly ask it to output ONLY 0 or 1 to make parsing easy.
    """
    messages = [
        {"role": "system", "content": "You are an expert multilingual linguist and sociologist. Your task is to analyze text for 'polarization'. Polarization involves sharp, hostile division, stereotyping out-groups, or implicit divisive framing. If the text is polarized, output ONLY the number '1'. If it is neutral or objective, output ONLY the number '0'. Do not explain your reasoning."},
        {"role": "user", "content": f"Text: {text}\n\nIs this text polarized? Output 1 for Yes, 0 for No."}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def parse_response(response_text):
    """Extracts the 0 or 1 from the LLM's output string."""
    # Llama-3 sometimes adds extra spaces or a period. We look for '1' or '0'.
    if '1' in response_text:
        return 1
    return 0 # Default to neutral if confused

# ==========================================
# 4. RUNNING EVALUATION ON THE DEV SET
# ==========================================
def evaluate_baseline():
    print(f"\n🔍 Reading Development Data from: {DEV_DIR}")
    files = glob.glob(f"{DEV_DIR}/*.csv")
    if not files:
        print("❌ No CSV files found! Check your Google Drive path.")
        return

    all_y_true = []
    all_y_pred = []

    for file in files:
        lang = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file)
        
        print(f"Processing {lang.upper()} ({len(df)} samples)...")
        
        lang_y_true = df['polarization'].tolist()
        lang_y_pred = []
        
        for idx, text in enumerate(df['text']):
            prompt = build_prompt(text)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5, # We only need 1 token (0 or 1)
                    temperature=0.1,  # Low temperature for deterministic answers
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode only the newly generated tokens (ignore the prompt)
            generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            prediction = parse_response(response)
            lang_y_pred.append(prediction)
            
            # Optional: Print first 2 examples to ensure it's working
            if idx < 2:
                print(f"   [Sample] Pred: {prediction} | Gold: {lang_y_true[idx]}")

        # Accumulate for overall score
        all_y_true.extend(lang_y_true)
        all_y_pred.extend(lang_y_pred)
        
        # Print per-language F1
        lang_f1 = f1_score(lang_y_true, lang_y_pred, average="macro")
        print(f"   ✅ {lang.upper()} Macro-F1: {lang_f1:.4f}\n")

    # Final Overall Score
    overall_f1 = f1_score(all_y_true, all_y_pred, average="macro")
    print("==========================================")
    print(f"🏆 LLAMA-3 ZERO-SHOT OVERALL MACRO-F1: {overall_f1:.4f}")
    print("==========================================")

# Run the evaluation
evaluate_baseline()