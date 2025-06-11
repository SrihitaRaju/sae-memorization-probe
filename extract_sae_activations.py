#!/usr/bin/env python3
"""
Extract SAE activations from a language model for memorization analysis.
"""

import argparse, os, gc, time, json, hashlib, logging, traceback, math
from glob import glob
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
import torch, torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file
from datasets import load_dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ───── SAE thin wrapper ──────────────────────────────────────────────
class SAE(nn.Module):
    def __init__(self, repo: str, layer: int, device: torch.device):
        super().__init__()
        # locate checkpoint
        if os.path.isdir(repo):
            pat = os.path.join(repo, f"2.8b_l{layer}", "checkpoints", "ae_*.*")
            files = sorted(glob(pat))
        else:
            prefix = f"2.8b_l{layer}/checkpoints/ae_"
            files  = sorted(f for f in list_repo_files(repo)
                            if f.startswith(prefix) and f.split(".")[-1] in ("pt","safetensors"))
        if not files:
            raise FileNotFoundError(f"No SAE checkpoint for L{layer}")
        ckpt = files[-1]; log.info(f"SAE ckpt: {ckpt}")
        local = ckpt if os.path.isfile(ckpt) else hf_hub_download(repo, ckpt)
        state = load_file(local) if local.endswith(".safetensors") else torch.load(local, map_location="cpu")
        state = state.get("state_dict", state)
        W = next(v for k,v in state.items() if "enc" in k and "weight" in k)
        b = next(v for k,v in state.items() if "enc" in k and "bias"  in k)
        self.W, self.b = W.to(device, torch.float16), b.to(device, torch.float16)
        self.n_features = self.W.shape[0]

    @torch.no_grad()
    def encode(self, h: torch.Tensor) -> torch.Tensor:
        B,S,D = h.shape
        return torch.relu(h.view(-1,D) @ self.W.T + self.b).view(B,S,self.n_features)

# ───── helpers ───────────────────────────────────────────────────────
def build_schema(n_feat, topk, save_last):
    fields = [
        ("idx", pa.int32()), ("label", pa.int8()), ("layer", pa.int8()),
        ("pool", pa.string()), ("features_dense", pa.binary(n_feat*2)),
        ("dup_count", pa.int64()), ("prefix_nll", pa.float32()),
        ("rare_rate", pa.float32()), ("domain", pa.string()),
        ("text_hash", pa.int64()),
    ]
    if topk>0:     fields.append(("features_sparse", pa.binary()))
    if save_last:  fields.append(("last_sparse", pa.binary()))
    return pa.schema(fields)

def flush_parquet(rows, schema, out_dir, layer, shard_idx):
    out = out_dir/f"L{layer}_{shard_idx:03d}.parquet"
    with pq.ParquetWriter(out, schema, compression="zstd") as w:
        w.write_table(pa.Table.from_pylist(rows, schema=schema))
    log.info(f"Wrote {out.name}  ({len(rows)} rows)")
    return shard_idx+1

def env_setup():
    for d in ("/dev/shm/hf_cache","/tmp/arrow_tmp"):
        os.makedirs(d, exist_ok=True)
    for k in ("HF_HOME","TRANSFORMERS_CACHE","TORCH_HOME","HF_DATASETS_CACHE"):
        os.environ.setdefault(k, "/dev/shm/hf_cache")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","max_split_size_mb:128")

# ───── main orchestration ───────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--sae_repo", required=True)
    p.add_argument("--layers", nargs="+", type=int, required=True)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--prefix_tok", type=int, default=96)
    p.add_argument("--add_special_tokens", action="store_true")
    p.add_argument("--pool", choices=["mean","max","both"], default="both")
    p.add_argument("--topk", type=int, default=128)
    p.add_argument("--save_last_sparse", action="store_true")
    p.add_argument("--shard", type=int, default=1000)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    env_setup()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ── checkpoint ──
    ckpt_path = out_dir/"checkpoint.json"
    ckpt = json.load(open(ckpt_path)) if (args.resume and ckpt_path.exists()) else {"done_layers":[]}
    def save_ckpt(): json.dump(ckpt, open(ckpt_path,"w"), indent=2)

    # ── dataset count ──
    with open(args.data) as f: total = sum(1 for _ in f)
    log.info(f"Rows: {total}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map={"": args.device}, low_cpu_mem_usage=True).eval()

    start = time.time()
    for layer in args.layers:
        if layer in ckpt["done_layers"]:
            log.info(f"[skip] L{layer}")
            continue

        sae = SAE(args.sae_repo, layer, torch.device(args.device))
        schema = build_schema(sae.n_features, args.topk, args.save_last_sparse)

        ds = load_dataset("json", data_files=args.data, streaming=True)["train"]
        rows, shard_idx = [], 0
        buf_ids, buf_lab, buf_idx, buf_meta = [], [], [], []

        for idx, ex in enumerate(tqdm(ds, total=total, desc=f"L{layer}")):
            ids = tok(ex["text"], add_special_tokens=args.add_special_tokens,
                      return_attention_mask=False).input_ids[:args.prefix_tok]
            if not ids: continue
            buf_ids.append(ids)
            buf_lab.append(1 if ex["label"]=="mem" else 0)
            buf_idx.append(idx)
            buf_meta.append(ex)

            if len(buf_ids) < args.batch: continue

            rows, shard_idx = process_batch(
                buf_ids, buf_lab, buf_idx, buf_meta,
                tok, model, sae, layer, args, schema, rows, shard_idx, out_dir
            )
            buf_ids, buf_lab, buf_idx, buf_meta = [], [], [], []

        # tail batch
        if buf_ids:
            rows, shard_idx = process_batch(
                buf_ids, buf_lab, buf_idx, buf_meta,
                tok, model, sae, layer, args, schema, rows, shard_idx, out_dir
            )
        if rows:
            shard_idx = flush_parquet(rows, schema, out_dir, layer, shard_idx)

        del sae; torch.cuda.empty_cache(); gc.collect()
        ckpt["done_layers"].append(layer); save_ckpt()
        log.info(f"L{layer} done  {(time.time()-start)/60:.1f} min elapsed")

    log.info(f"✓ All complete in {(time.time()-start)/60:.1f} min")

# ───── batch processing ─────────────────────────────────────────────
def get_mlp_out_hook(layer_idx, store):
    """
    Returns a forward hook that captures the output of
    gpt_neox.layers[layer_idx].mlp (after dense_4h_to_h, before residual add).
    """
    def _hook(module, inp, out):
        store["h"] = out            # (B, S, d_model)
    return _hook
def process_batch(ids_buf, lab_buf, idx_buf, meta_buf,
                  tok, model, sae, layer, args, schema, rows, shard_idx, out_dir):
    try:
        max_len = max(len(x) for x in ids_buf)
        pad_id  = tok.pad_token_id
        padded = [x + [pad_id]*(max_len-len(x)) for x in ids_buf]
        mask   = [[1]*len(x)+[0]*(max_len-len(x)) for x in ids_buf]

        toks = {
            'input_ids': torch.tensor(padded, dtype=torch.long, device=args.device),
            'attention_mask': torch.tensor(mask, dtype=torch.long, device=args.device)
        }
        store = {}
        handle = model.gpt_neox.layers[layer].mlp.register_forward_hook(
             get_mlp_out_hook(layer, store)
         )
        with torch.no_grad():
            _ = model(**toks, use_cache=False)   # no need for hidden_states now
        h = store["h"]                       # ← correct MLP_out
        handle.remove()
        z = sae.encode(h)                                       # (B,S,F)

        m = toks['attention_mask'].unsqueeze(-1)
        z_masked = z * m

        pools = {}
        if args.pool in ("mean","both"):
            pools["mean"] = z_masked.sum(1) / m.sum(1)
        if args.pool in ("max","both"):
            pools["max"]  = (z_masked + (m==0)*-1e4).max(1).values

        if args.save_last_sparse:
            last_pos = m.squeeze(-1).sum(1) - 1
            last_vec = z[torch.arange(z.size(0)), last_pos]

        for b in range(len(ids_buf)):
            ex = meta_buf[b]
            hsh = int(hashlib.md5(" ".join(map(str,ids_buf[b])).encode()).hexdigest()[:15], 16)
            for pname, vec in pools.items():
                row = dict(
                    idx=idx_buf[b], label=lab_buf[b], layer=layer,
                    pool=pname, features_dense=vec[b].half().cpu().numpy().tobytes(),
                    dup_count=ex.get("dup_count"), prefix_nll=ex.get("prefix_nll"),
                    rare_rate=ex.get("rare_rate"), domain=ex.get("domain"),
                    text_hash=hsh,
                )
                if args.topk>0:
                    v, i = torch.topk(vec[b], args.topk)
                    row["features_sparse"] = (
                        i.cpu().numpy().astype(np.uint16).tobytes() +
                        v.cpu().numpy().astype(np.float16).tobytes()
                    )
                if args.save_last_sparse:
                    nz = last_vec[b] > 0
                    if nz.any():
                        li = torch.nonzero(nz, as_tuple=False).squeeze(-1).cpu().numpy().astype(np.uint16)
                        lv = last_vec[b][nz].cpu().numpy().astype(np.float16)
                        row["last_sparse"] = li.tobytes() + lv.tobytes()
                rows.append(row)

        # shard flush
        if len(rows) >= args.shard:
            shard_idx = flush_parquet(rows, schema, out_dir, layer, shard_idx)
            rows.clear(); torch.cuda.empty_cache(); gc.collect()

    except Exception:
        log.error("Batch failed:\n" + traceback.format_exc())
        torch.cuda.empty_cache()
    return rows, shard_idx

if __name__ == "__main__":
    main()
