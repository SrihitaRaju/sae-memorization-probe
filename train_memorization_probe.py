#!/usr/bin/env python3
"""
Train a neural network probe to detect memorization from SAE activations.
"""
import argparse, time, numpy as np, torch, pyarrow as pa, pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

torch.backends.cuda.matmul.allow_tf32 = True  # faster FP32 matmul

# ───────────────────────── Arrow ⇒ NumPy ──────────────────────────
def decode_fixed(col: pa.FixedSizeBinaryArray) -> np.ndarray:
    col = col.combine_chunks()
    buf = memoryview(col.buffers()[1])
    n, row_b = len(col), col.type.byte_width
    return np.frombuffer(buf, dtype=np.float16).reshape(n, row_b // 2)

# ───────────────────────── Data loader ────────────────────────────
def load_balanced(path, layer, pool, feat_col, n_each):
    shards = sorted(Path(path).glob(f"L{layer}_*.parquet"))
    pos, neg = [], []
    for fp in tqdm(shards, desc=f"layer{layer}", unit="shard"):
        tbl = pq.read_table(fp, columns=[feat_col, "label", "pool"])
        tbl = tbl.filter(pc.equal(tbl["pool"], pa.scalar(pool)))
        if len(tbl) == 0:
            continue
        feats = decode_fixed(tbl[feat_col]).astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        labels = tbl["label"].to_numpy(zero_copy_only=False)
        need_p, need_n = n_each - len(pos), n_each - len(neg)
        if need_p: pos.extend(feats[labels == 1][:need_p])
        if need_n: neg.extend(feats[labels == 0][:need_n])
        if len(pos) >= n_each and len(neg) >= n_each:
            break
    X = np.vstack([pos[:n_each], neg[:n_each]])
    y = np.hstack([np.ones(n_each), np.zeros(n_each)])
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

# ───────────────────────── Simple MLP probe ───────────────────────
def build_mlp(d_in, hid):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, hid),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(hid, 1),
    )

def run_mlp(X, y, hid, epochs, batch, scale_mode, lr, clip_val):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, stratify=y, test_size=0.20, random_state=0)

    if scale_mode == "standard":
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_va = scaler.transform(X_tr), scaler.transform(X_va)
    elif scale_mode == "maxabs":
        scaler = MaxAbsScaler().fit(X_tr)
        X_tr, X_va = scaler.transform(X_tr), scaler.transform(X_va)

    X_tr = torch.tensor(X_tr, dtype=torch.float32, device=dev)
    y_tr = torch.tensor(y_tr, dtype=torch.float32, device=dev)
    X_va = torch.tensor(X_va, dtype=torch.float32, device=dev)
    y_va = torch.tensor(y_va, dtype=torch.float32, device=dev)

    net = build_mlp(X_tr.shape[1], hid).to(dev).float()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-4)
    best = 0.0
    for ep in range(1, epochs + 1):
        t0 = time.time(); net.train()
        for idx in torch.randperm(len(y_tr), device=dev).split(batch):
            xb, yb = X_tr[idx], y_tr[idx]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                net(xb).squeeze(-1), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_val)
            opt.step()
        net.eval()
        with torch.no_grad():
            preds = torch.sigmoid(net(X_va).squeeze(-1)).cpu().numpy()
        auc = roc_auc_score(y_va.cpu().numpy(), preds)
        best = max(best, auc)
        print(f"Ep{ep:02d} loss {loss.item():.3f} valAUC {auc:.3f} {time.time()-t0:.1f}s")
    print(f"Best val AUC {best:.3f}")

# ───────────────────────── CLI + auto-cache ───────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--feat_col", default="features_dense")
    ap.add_argument("--n_each", type=int, default=5000)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--scale", choices=["standard", "maxabs", "none"],
                    default="none")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.5)
    args = ap.parse_args()

    # ---------- auto-cache ----------
    cache_dir = Path("cache"); cache_dir.mkdir(exist_ok=True)
    tag = f"l{args.layer}_{args.feat_col}_{args.n_each}"
    fX = cache_dir / f"{tag}.npy"
    fy = cache_dir / f"{tag}_y.npy"

    if fX.exists() and fy.exists():
        t0 = time.time()
        X = np.load(fX); y = np.load(fy)
        print(f"[cache] loaded {tag} in {time.time()-t0:.1f}s | X {X.shape}")
    else:
        print("[cache] miss – decoding Parquet …")
        t0 = time.time()
        X, y = load_balanced(args.parquet_dir, args.layer,
                             "mean", args.feat_col, args.n_each)
        np.save(fX, X); np.save(fy, y)
        print(f"[cache] saved to {fX} ({time.time()-t0:.1f}s)")

    run_mlp(X, y, args.hidden, args.epochs,
            args.batch, args.scale, args.lr, args.clip)
