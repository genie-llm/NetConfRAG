"""
baselines/llm_only.py — Baseline 1: LLM only (no retrieval).

Runs the Llama model directly on each question from config.INPUT_CSV and writes
the cleaned Cisco IOS answer to config.OUTPUT_CSV column "llm_only".

Usage:
    python baselines/llm_only.py
"""

import os
import sys
import time

import torch
import pandas as pd
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Allow importing config from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from utils import extract_cisco_config


# ── LLM ───────────────────────────────────────────────────────────────────────

def load_llm():
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading LLM: {config.LLM_MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✅ LLM loaded.")
    return model, tokenizer


def ask_model(question: str, model, tokenizer) -> str:
    """Direct LLM answer — no retrieval context."""
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a Cisco network engineer. Generate ONLY valid Cisco IOS commands "
        "in a single code block.\n\n"
        "Rules:\n"
        "- Start with: enable, configure terminal\n"
        "- End with: end\n"
        "- Use real Cisco commands only\n"
        "- No explanations or comments\n"
        "- No multiple choice answers\n\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}\n\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n```"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            temperature=0.1,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    if answer.startswith("```"):
        answer = answer[3:]
    if answer.endswith("```"):
        answer = answer[:-3]
    return f"```\n{answer.strip()}\n```"


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    login(config.HF_TOKEN)
    model, tokenizer = load_llm()

    if not os.path.exists(config.INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {config.INPUT_CSV}")
    df = pd.read_csv(config.INPUT_CSV)
    if "question" not in df.columns:
        raise ValueError("Input CSV must contain a 'question' column.")

    TARGET_COL = "llm_only"
    if TARGET_COL not in df.columns:
        df[TARGET_COL] = ""

    rows_to_process = df.index[df[TARGET_COL].astype(str).str.strip() == ""].tolist()
    print(f"Total rows: {len(df)} | To generate: {len(rows_to_process)}\n")

    output_csv = config.OUTPUT_CSV.replace(".csv", "_llm_only.csv")

    for i in rows_to_process:
        q = str(df.at[i, "question"]).strip()
        if not q:
            df.at[i, TARGET_COL] = ""
            continue
        try:
            raw     = ask_model(q, model, tokenizer)
            cleaned = extract_cisco_config(raw)
            df.at[i, TARGET_COL] = cleaned
            print(f"[{i+1}/{len(df)}] ✅ {q[:60]}")
        except KeyboardInterrupt:
            print("\nInterrupted. Saving progress…")
            break
        except Exception as e:
            print(f"[{i+1}/{len(df)}] ⚠️  Failed (row {i}): {e}")

        df.to_csv(output_csv, index=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if config.SLEEP_BETWEEN_CALLS > 0:
            time.sleep(config.SLEEP_BETWEEN_CALLS)

    df.to_csv(output_csv, index=False)
    print(f"\n✅ Done → {output_csv}")


if __name__ == "__main__":
    main()
