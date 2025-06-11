#!/usr/bin/env python
"""
Detect memorization in language model outputs by comparing generated text with ground truth.
"""

import argparse, json, os, random, gc
from pathlib import Path
import time

import torch, datasets, tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def env_setup():
    """Optimized for cloud GPU environments"""
    # Determine cache directory first
    cache_dir = "/dev/shm/hf_cache" if os.path.exists("/dev/shm") else "/tmp/hf_cache"
    
    cache_dirs = [cache_dir, "/tmp"]
    for d in cache_dirs:
        os.makedirs(d, exist_ok=True)
    
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    os.environ.setdefault("TORCH_HOME", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", 
                          "max_split_size_mb:128,garbage_collection_threshold:0.6")
    
    print(f"[INFO] Using cache directory: {cache_dir}")


def count_exact_prefix_match(gen_ids, tgt_ids):
    """Count consecutive exact matches from the beginning"""
    cnt = 0
    for g, t in zip(gen_ids, tgt_ids):
        if g == t:
            cnt += 1
        else:
            break
    return cnt


@torch.inference_mode()
def sweep_matches(texts, tok, model, prefix_len, gen_len, batch_size):
    """Clean batch processing without padding complexity"""
    ratios, consec = [], []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"[INFO] Processing {len(texts)} sequences in {total_batches} batches of size {batch_size}")
    
    for batch_idx in tqdm.trange(0, len(texts), batch_size, desc="Memorization check"):
        batch_txt = texts[batch_idx:batch_idx + batch_size]
        
        try:
            # Simple tokenization - no padding needed for uniform 200-token sequences
            enc = tok(batch_txt, 
                     return_tensors="pt",
                     add_special_tokens=False, 
                     padding=False,  # No padding needed!
                     truncation=False).to(model.device)
            
            ids_full = enc.input_ids
            
            # Verify sequence lengths
            if ids_full.shape[1] < prefix_len + gen_len:
                print(f"[WARNING] Batch {batch_idx//batch_size} has sequences too short ({ids_full.shape[1]} tokens), skipping")
                ratios.extend([0.0] * len(batch_txt))
                consec.extend([0] * len(batch_txt))
                continue
            
            # Simple indexing - extract prefix and target from beginning
            prefix_ids = ids_full[:, :prefix_len]
            target_ids = ids_full[:, prefix_len:prefix_len + gen_len]
            
            # Generate continuation
            gen = model.generate(
                prefix_ids,
                max_new_tokens=gen_len,
                min_new_tokens=gen_len,  # Force exact generation length
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,  # Prevent infinite generation
                use_cache=True,
            )
            
            # Verify generation length
            if gen.shape[1] < prefix_len + gen_len:
                actual_gen = gen.shape[1] - prefix_len
                print(f"[WARNING] Generated only {actual_gen} tokens, expected {gen_len}")
                # Pad the results for this batch
                ratios.extend([0.0] * len(batch_txt))
                consec.extend([0] * len(batch_txt))
                continue
            
            # Extract generated tokens (skip the input prefix)
            gen_suffix = gen[:, prefix_len:prefix_len + gen_len]
            
            # Calculate matches
            match_tok = (gen_suffix == target_ids)
            batch_ratios = match_tok.float().mean(dim=1).cpu().tolist()
            
            # Calculate consecutive matches
            batch_consec = [
                count_exact_prefix_match(g.tolist(), t.tolist())
                for g, t in zip(gen_suffix, target_ids)
            ]
            
            ratios.extend(batch_ratios)
            consec.extend(batch_consec)
            
            # Memory cleanup every 10 batches
            if (batch_idx // batch_size) % 10 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"[ERROR] Batch {batch_idx//batch_size} failed: {e}")
            # Add zero scores for failed batch
            ratios.extend([0.0] * len(batch_txt))
            consec.extend([0] * len(batch_txt))
            continue
    
    return ratios, consec


def main():
    ap = argparse.ArgumentParser(description="Clean memorization detection for uniform-length sequences")
    ap.add_argument("jsonl_in", help="Input JSONL file with text and duplication_count")
    ap.add_argument("--out", default="mem_vs_nonmem.jsonl", help="Output JSONL file")
    ap.add_argument("--prefix", type=int, default=32, help="Prefix length in tokens")
    ap.add_argument("--gen", type=int, default=64, help="Generation length in tokens")
    ap.add_argument("--pos_thr", type=float, default=0.8, help="Positive threshold for memorization")
    ap.add_argument("--neg_thr", type=float, default=0.1, help="Negative threshold for non-memorization")
    ap.add_argument("--n_each", type=int, default=1000, help="Number of examples per class")
    ap.add_argument("--batch", type=int, default=16, help="Batch size")
    ap.add_argument("--max_total", type=int, default=None, help="Max sequences to process")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    start_time = time.time()
    rng = random.Random(args.seed)
    env_setup()
    
    print(f"[INFO] GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU device: {torch.cuda.get_device_name()}")
        print(f"[INFO] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # 1 ─ Load corpus
    print("[INFO] Loading dataset...")
    ds = datasets.Dataset.from_json(args.jsonl_in)
    texts = ds["text"]
    dup_counts = ds["duplication_count"] if "duplication_count" in ds.column_names else [None] * len(texts)

    if args.max_total is not None:
        texts = texts[:args.max_total]
        dup_counts = dup_counts[:args.max_total]
        print(f"[INFO] Truncated to first {len(texts):,} sequences")

    # 2 ─ Load model 
    print("[INFO] Loading model and tokenizer...")
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
    
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    
    print(f"[INFO] Pad token: {tok.pad_token} (ID: {tok.pad_token_id})")
    print(f"[INFO] No padding needed - uniform sequence lengths")
    
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-2.8b",
        torch_dtype=torch.float16,
        device_map={"": 0},  # Force everything on GPU 0 for single-GPU efficiency
        low_cpu_mem_usage=True,
    ).eval()

    # 3 ─ Quick validation
    print("[INFO] Validating sequence lengths...")
    sample_lengths = [len(tok.encode(t, add_special_tokens=False)) for t in texts[:10]]
    print(f"[INFO] Sample lengths: {sample_lengths}")
    print(f"[INFO] Required: prefix({args.prefix}) + gen({args.gen}) = {args.prefix + args.gen} tokens")

    # 4 ─ Evaluate memorization
    print("[INFO] Starting memorization detection...")
    ratios, consec = sweep_matches(
        texts, tok, model,
        prefix_len=args.prefix,
        gen_len=args.gen,
        batch_size=args.batch,
    )

    # 5 ─ Classify results
    positives, negatives = [], []
    for txt, dup, r, c in zip(texts, dup_counts, ratios, consec):
        row = {
            "text": txt,
            "dup_count": dup,
            "match_ratio": r,
            "match_consec": c
        }
        if r >= args.pos_thr:
            positives.append(row)
        elif r <= args.neg_thr:
            negatives.append(row)

    print(f"\n[RESULTS] Memorized (≥{args.pos_thr}): {len(positives):,}")
    print(f"[RESULTS] Non-memorized (≤{args.neg_thr}): {len(negatives):,}")
    print(f"[RESULTS] Ambiguous: {len(texts) - len(positives) - len(negatives):,}")
    
    # Print detailed statistics
    if positives:
        pos_ratios = [p["match_ratio"] for p in positives]
        pos_consec = [p["match_consec"] for p in positives]
        pos_dups = [p["dup_count"] for p in positives if p["dup_count"] is not None]
        avg_dup_str = f"{sum(pos_dups)/len(pos_dups):.0f}" if pos_dups else "N/A"
        print(f"[STATS] Memorized - ratio: {sum(pos_ratios)/len(pos_ratios):.3f}, consec: {sum(pos_consec)/len(pos_consec):.1f}, avg_dup: {avg_dup_str}")
        
    if negatives:
        neg_ratios = [n["match_ratio"] for n in negatives]
        neg_consec = [n["match_consec"] for n in negatives]
        neg_dups = [n["dup_count"] for n in negatives if n["dup_count"] is not None]
        avg_dup_str = f"{sum(neg_dups)/len(neg_dups):.0f}" if neg_dups else "N/A"
        print(f"[STATS] Non-memorized - ratio: {sum(neg_ratios)/len(neg_ratios):.3f}, consec: {sum(neg_consec)/len(neg_consec):.1f}, avg_dup: {avg_dup_str}")

    # Check sufficiency and sample results
    n_pos = min(len(positives), args.n_each) if positives else 0
    n_neg = min(len(negatives), args.n_each) if negatives else 0
    
    if n_pos < args.n_each:
        print(f"\n[WARNING] Only {n_pos} memorized examples available (need {args.n_each})")
        if positives:
            print(f"[SUGGESTION] Try: --pos_thr {max(0.1, args.pos_thr - 0.2):.1f}")
    
    if n_neg < args.n_each:
        print(f"[WARNING] Only {n_neg} non-memorized examples available (need {args.n_each})")
        if negatives:
            print(f"[SUGGESTION] Try: --neg_thr {min(0.9, args.neg_thr + 0.2):.1f}")

    keep_pos = rng.sample(positives, n_pos) if n_pos > 0 else []
    keep_neg = rng.sample(negatives, n_neg) if n_neg > 0 else []

    # 6 ─ Write results
    out = Path(args.out)
    with out.open("w") as fh:
        for row in keep_pos:
            json.dump({"label": "mem", **row}, fh, ensure_ascii=False)
            fh.write("\n")
        for row in keep_neg:
            json.dump({"label": "nonmem", **row}, fh, ensure_ascii=False)
            fh.write("\n")

    elapsed = time.time() - start_time
    print(f"\n[DONE] Wrote {len(keep_pos) + len(keep_neg):,} rows → {out.resolve()}")
    print(f"[DONE] Processing time: {elapsed/60:.1f} minutes")
    print(f"[FINAL] Memorized: {len(keep_pos)}, Non-memorized: {len(keep_neg)}")


if __name__ == "__main__":
    main()