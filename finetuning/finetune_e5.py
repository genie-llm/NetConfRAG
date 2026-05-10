#!/usr/bin/env python3
"""
finetune_e5.py  — fine-tuning of intfloat/e5-small-v2 on Cisco RAG triplets.

- Saves models natively in SentenceTransformer/HuggingFace format directly
  in the checkpoint root (no nested folders).
- Generates `metadata.json` with `checkpoint_type` and `triplet_accuracy` 
  so `build_vectorstore.py` can automatically discover and score checkpoints.
- Emits `sentence_bert_config.json` sentinel for Langchain compatibility.

Usage:
  # Full training run
  python3 finetune_e5.py --triplets triplets.jsonl --output-dir ./e5-finetuned

  # Pre-split files
  python3 finetune_e5.py --train triplets_train.jsonl --val triplets_val.jsonl --output-dir ./e5-finetuned
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL      = "intfloat/e5-small-v2"
MAX_SEQ_LEN     = 512       # e5-small-v2 supports up to 512 tokens
TRAIN_VAL_RATIO = 0.9       # used when --triplets is given (no pre-split)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning(f"Line {i} parse error in {path}: {e}")
    return records


def split_by_topic(
    records: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Stratified train/val split by topic (_meta.topic)."""
    rng = random.Random(seed)
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        topic = r.get("_meta", {}).get("topic", "unknown")
        by_topic[topic].append(r)

    train, val = [], []
    for topic, recs in by_topic.items():
        rng.shuffle(recs)
        n_val = max(1, int(len(recs) * val_ratio))
        val.extend(recs[:n_val])
        train.extend(recs[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class TripletDataset(Dataset):
    def __init__(self, records: list[dict], score_threshold: int = 2):
        self.records = [
            r for r in records
            if r.get("_meta", {}).get("eval_score", 3) >= score_threshold
            and r.get("_meta", {}).get("eval_detail", {}).get("A", 1) == 1
        ]
        log.info(
            f"Dataset: {len(records)} records → {len(self.records)} after "
            f"score/A-check filter (threshold={score_threshold})"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        return {
            "query": r["query"],   # already prefixed "query: ..."
            "pos":   r["pos"],     # already prefixed "passage: ..."
            "neg":   r["neg"],     # already prefixed "passage: ..."
        }


@dataclass
class TripletCollator:
    tokenizer: object
    max_length: int = MAX_SEQ_LEN

    def __call__(self, batch: list[dict]) -> dict[str, dict]:
        queries   = [b["query"] for b in batch]
        positives = [b["pos"]  for b in batch]
        negatives = [b["neg"]  for b in batch]

        def tok(texts):
            return self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        return {
            "query": tok(queries),
            "pos":   tok(positives),
            "neg":   tok(negatives),
        }


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class E5Model(torch.nn.Module):
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)

    def _pool(self, model_output, attention_mask: Tensor) -> Tensor:
        token_emb = model_output.last_hidden_state        # (B, T, H)
        mask_exp  = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        sum_emb   = (token_emb * mask_exp).sum(dim=1)     # (B, H)
        count     = mask_exp.sum(dim=1).clamp(min=1e-9)   # (B, 1)
        return F.normalize(sum_emb / count, p=2, dim=-1)  # (B, H)

    def encode(self, batch: dict) -> Tensor:
        out = self.encoder(**batch)
        return self._pool(out, batch["attention_mask"])

    def forward(
        self,
        query_batch: dict,
        pos_batch: dict,
        neg_batch: dict,
        temperature: float = 0.02,
    ) -> tuple[Tensor, dict]:
        q_emb   = self.encode(query_batch)
        pos_emb = self.encode(pos_batch)
        neg_emb = self.encode(neg_batch)

        scores = torch.matmul(q_emb, pos_emb.T) / temperature
        neg_scores = (q_emb * neg_emb).sum(dim=-1, keepdim=True) / temperature
        all_scores = torch.cat([scores, neg_scores], dim=1)
        labels = torch.arange(len(q_emb), device=q_emb.device)
        loss_inbatch = F.cross_entropy(all_scores, labels)

        sim_pos = (q_emb * pos_emb).sum(dim=-1)
        sim_neg = (q_emb * neg_emb).sum(dim=-1)
        loss_margin = F.relu(0.05 - sim_pos + sim_neg).mean()

        loss = loss_inbatch + loss_margin

        metrics = {
            "loss":         loss.item(),
            "loss_inbatch": loss_inbatch.item(),
            "loss_margin":  loss_margin.item(),
            "sim_pos_mean": sim_pos.mean().item(),
            "sim_neg_mean": sim_neg.mean().item(),
            "sim_gap":      (sim_pos - sim_neg).mean().item(),
        }
        return loss, metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: E5Model,
    val_loader: DataLoader,
    device: torch.device,
    temperature: float = 0.02,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    sim_pos_all, sim_neg_all = [], []

    for batch in val_loader:
        qb = {k: v.to(device) for k, v in batch["query"].items()}
        pb = {k: v.to(device) for k, v in batch["pos"].items()}
        nb = {k: v.to(device) for k, v in batch["neg"].items()}

        loss, metrics = model(qb, pb, nb, temperature=temperature)
        total_loss += loss.item()
        total_samples += 1

        q_emb   = model.encode(qb)
        pos_emb = model.encode(pb)
        neg_emb = model.encode(nb)
        sim_pos_all.append((q_emb * pos_emb).sum(dim=-1).mean().item())
        sim_neg_all.append((q_emb * neg_emb).sum(dim=-1).mean().item())

    n = max(1, total_samples)
    return {
        "val_loss":     total_loss / n,
        "val_sim_pos":  sum(sim_pos_all) / n,
        "val_sim_neg":  sum(sim_neg_all) / n,
        "val_sim_gap":  (sum(sim_pos_all) - sum(sim_neg_all)) / n,
    }


@torch.no_grad()
def evaluate_accuracy(
    model: E5Model,
    val_records: list[dict],
    tokenizer,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    model.eval()
    correct = 0
    total   = 0
    collator = TripletCollator(tokenizer)

    for i in range(0, len(val_records), batch_size):
        chunk = val_records[i:i + batch_size]
        batch = collator(
            [{"query": r["query"], "pos": r["pos"], "neg": r["neg"]} for r in chunk]
        )
        qb = {k: v.to(device) for k, v in batch["query"].items()}
        pb = {k: v.to(device) for k, v in batch["pos"].items()}
        nb = {k: v.to(device) for k, v in batch["neg"].items()}

        q_emb   = model.encode(qb)
        pos_emb = model.encode(pb)
        neg_emb = model.encode(nb)
        sp = (q_emb * pos_emb).sum(dim=-1)
        sn = (q_emb * neg_emb).sum(dim=-1)
        correct += (sp > sn).float().sum().item()
        total   += len(chunk)

    return {"acc@1": correct / max(1, total), "n": total}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    output_dir:      str   = "./e5-finetuned"
    epochs:          int   = 2
    batch_size:      int   = 32
    grad_accum:      int   = 2
    lr:              float = 2e-6
    warmup_ratio:    float = 0.06
    weight_decay:    float = 0.01
    temperature:     float = 0.02
    max_grad_norm:   float = 1.0
    eval_every:      int   = 200
    save_every:      int   = 400
    score_threshold: int   = 2
    val_ratio:       float = 0.1
    seed:            int   = 42
    smoke_test:      bool  = False
    fp16:            bool  = False
    bf16:            bool  = False


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def train(cfg: TrainConfig, train_records: list[dict], val_records: list[dict]):
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    train_ds = TripletDataset(train_records, score_threshold=cfg.score_threshold)
    val_ds   = TripletDataset(val_records,   score_threshold=cfg.score_threshold)

    if cfg.smoke_test:
        train_ds.records = train_ds.records[:cfg.batch_size * 4]
        val_ds.records   = val_ds.records[:cfg.batch_size * 2]
        cfg.epochs, cfg.eval_every, cfg.save_every = 1, 5, 10
        log.info("Smoke test mode: truncated dataset, 1 epoch")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    collator  = TripletCollator(tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collator, num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        collate_fn=collator, num_workers=2, pin_memory=(device.type == "cuda"),
    )

    model = E5Model(BASE_MODEL).to(device)

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(params, lr=cfg.lr)
    except ImportError:
        optimizer = torch.optim.AdamW(params, lr=cfg.lr)

    steps_per_epoch = len(train_loader) // cfg.grad_accum
    total_steps     = steps_per_epoch * cfg.epochs
    warmup_steps    = int(total_steps * cfg.warmup_ratio)
    scheduler       = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp   = (device.type == "cuda") and (cfg.fp16 or cfg.bf16)
    amp_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp and cfg.fp16)

    global_step  = 0
    best_val_gap = -float("inf")
    last_val_acc = 0.0

    def save_checkpoint(tag: str, state: dict, acc: float = 0.0):
        ckpt_dir = out_dir / f"checkpoint_{tag}"
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. Save standard SentenceTransformer format DIRECTLY in ckpt_dir
        _save_as_sentence_transformer(model, tokenizer, ckpt_dir)
        
        # 2. Save wrapper script state dict (for potential exact resume capabilities)
        torch.save(model.state_dict(), ckpt_dir / "model_wrapper.pt")
        torch.save(state, ckpt_dir / "trainer_state.pt")

        # 3. Create metadata.json for 02_build_vectorstore.py compatibility
        metadata = {
            "checkpoint_type": "best" if tag == "best" else "step",
            "metrics": {
                "triplet_accuracy": acc,
                "val_sim_gap": state.get("best_val_gap", 0.0)
            }
        }
        (ckpt_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        log.info(f"Saved checkpoint: {ckpt_dir} (Acc@1: {acc:.4f})")

    # ---- Resume ----
    resume_path = out_dir / "checkpoint_latest"
    if resume_path.exists():
        ckpt = torch.load(resume_path / "trainer_state.pt", map_location=device)
        model.load_state_dict(torch.load(resume_path / "model_wrapper.pt", map_location=device))
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step  = ckpt["global_step"]
        best_val_gap = ckpt.get("best_val_gap", best_val_gap)
        log.info(f"Resumed from step {global_step}")

    log_f = open(out_dir / "train_log.jsonl", "a", buffering=1)
    
    # ---- Training ----
    model.train()
    optimizer.zero_grad()
    step_loss = 0.0
    step_metrics = defaultdict(float)
    t0 = time.time()

    for epoch in range(cfg.epochs):
        for micro_step, batch in enumerate(train_loader):
            qb = move_batch(batch["query"], device)
            pb = move_batch(batch["pos"],   device)
            nb = move_batch(batch["neg"],   device)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                loss, metrics = model(qb, pb, nb, temperature=cfg.temperature)
                loss = loss / cfg.grad_accum

            if use_amp and cfg.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_loss += loss.item() * cfg.grad_accum
            for k, v in metrics.items():
                step_metrics[k] += v

            if (micro_step + 1) % cfg.grad_accum == 0:
                if use_amp and cfg.fp16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if use_amp and cfg.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    n = max(1, cfg.grad_accum)
                    avg = {k: v / n for k, v in step_metrics.items()}
                    log.info(
                        f"E{epoch+1} step={global_step} | loss={avg['loss']:.4f} | "
                        f"gap={avg['sim_gap']:.4f} | pos={avg['sim_pos_mean']:.4f} | "
                        f"lr={scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s"
                    )
                    log_f.write(json.dumps({
                        "step": global_step, "epoch": epoch + 1,
                        **{k: round(v, 6) for k, v in avg.items()},
                        "lr": scheduler.get_last_lr()[0],
                    }) + "\n")
                    step_loss = 0.0
                    step_metrics = defaultdict(float)

                if global_step % cfg.eval_every == 0:
                    acc_result  = evaluate_accuracy(model, val_ds.records, tokenizer, device, cfg.batch_size * 2)
                    val_metrics = evaluate(model, val_loader, device, cfg.temperature)
                    last_val_acc = acc_result['acc@1']

                    log.info(
                        f"[EVAL] step={global_step} | val_loss={val_metrics['val_loss']:.4f} | "
                        f"acc@1={last_val_acc:.4f} ({acc_result['n']} pairs) | gap={val_metrics['val_sim_gap']:.4f}"
                    )

                    if val_metrics["val_sim_gap"] > best_val_gap:
                        best_val_gap = val_metrics["val_sim_gap"]
                        save_checkpoint("best", {
                            "global_step": global_step,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "best_val_gap": best_val_gap,
                        }, acc=last_val_acc)
                        log.info(f"  ↑ New best val_sim_gap={best_val_gap:.4f}")
                    model.train()

                if global_step % cfg.save_every == 0:
                    state = {
                        "global_step": global_step,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_val_gap": best_val_gap,
                    }
                    save_checkpoint("latest", state, acc=last_val_acc)

                if cfg.smoke_test and global_step >= 10:
                    log_f.close()
                    return

    save_checkpoint("final", {
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_gap": best_val_gap,
    }, acc=last_val_acc)
    log_f.close()
    log.info(f"Training complete. Best val_sim_gap={best_val_gap:.4f}")

# ---------------------------------------------------------------------------
# Sentence-transformers export
# ---------------------------------------------------------------------------

def _save_as_sentence_transformer(model: E5Model, tokenizer, out_dir: Path):
    """
    Saves the HuggingFace base model and creates the specific configuration 
    files required for the standard SentenceTransformer integration to work natively.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save core huggingface model (creates pytorch_model.bin / model.safetensors & config.json)
        model.encoder.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        
        # 2. Save modules.json pointing to pooling component
        modules = [
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        ]
        (out_dir / "modules.json").write_text(json.dumps(modules, indent=2))
        
        # 3. Save Pooling configuration
        hidden_size = model.encoder.config.hidden_size
        pooling_config = {
            "word_embedding_dimension": hidden_size,
            "pooling_mode_cls_token": False,
            "pooling_mode_mean_tokens": True,
            "pooling_mode_max_tokens": False,
        }
        (out_dir / "1_Pooling").mkdir(exist_ok=True)
        (out_dir / "1_Pooling" / "config.json").write_text(json.dumps(pooling_config, indent=2))
        
        # 4. Save Sentinel File: needed by build_vectorstore.py and SentenceTransformers
        (out_dir / "sentence_bert_config.json").write_text(
            json.dumps({"max_seq_length": MAX_SEQ_LEN, "do_lower_case": False}, indent=2)
        )
        
    except Exception as e:
        log.warning(f"sentence-transformer export failed (non-fatal): {e}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune intfloat/e5-small-v2 (v2) on Cisco RAG triplets")
    parser.add_argument("--triplets", default=None, help="JSONL file to be split automatically")
    parser.add_argument("--train", default=None, help="Pre-split train JSONL")
    parser.add_argument("--val", default=None, help="Pre-split val JSONL")
    parser.add_argument("--val-ratio", type=float, default=TRAIN_VAL_RATIO)
    parser.add_argument("--score-threshold", type=int, default=2)
    parser.add_argument("--output-dir", default="./e5-cisco")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.triplets:
        records = load_jsonl(args.triplets)
        train_records, val_records = split_by_topic(records, val_ratio=1-args.val_ratio, seed=args.seed)
    elif args.train and args.val:
        train_records, val_records = load_jsonl(args.train), load_jsonl(args.val)
    else:
        train_records, val_records = [], []

    cfg = TrainConfig(
        output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size,
        grad_accum=args.grad_accum, lr=args.lr, warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay, temperature=args.temperature,
        eval_every=args.eval_every, save_every=args.save_every,
        score_threshold=args.score_threshold, val_ratio=1-args.val_ratio,
        seed=args.seed, smoke_test=args.smoke_test, fp16=args.fp16, bf16=args.bf16,
    )

    if args.eval_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = E5Model(BASE_MODEL).to(device)
        best_ckpt = Path(args.output_dir) / "checkpoint_best" / "model_wrapper.pt"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device))
        
        result = evaluate_accuracy(model, TripletDataset(val_records).records, tokenizer, device)
        log.info(f"Acc@1 = {result['acc@1']:.4f} over {result['n']} pairs")
        return

    train(cfg, train_records, val_records)

if __name__ == "__main__":
    main()
