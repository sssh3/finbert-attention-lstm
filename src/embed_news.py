"""
embed_news.py
-------------
Encode news headlines with FinBERT (ProsusAI/finbert).
Uses MPS (Apple Silicon) if available, else CPU.

Reads:  data/processed/features.parquet  (headline column)
Writes: data/processed/embeddings.npy    (N × 768 float32)
        data/processed/pca.pkl           (fitted sklearn PCA → 64-d)
        data/processed/embeddings_pca.npy (N × 64 float32, input-ready)

Rows in embeddings match rows in features.parquet exactly (same order).
Empty headlines (no-news days) → zero vector.

Usage:
    python src/embed_news.py
"""

import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME   = "ProsusAI/finbert"
BATCH_SIZE   = 64          # increase if MPS RAM allows
PCA_DIM      = 64
MAX_LENGTH   = 128         # headline text; rarely exceeds this
PROCESSED_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings, ignoring padding tokens."""
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def encode_headlines(headlines: list[str], tokenizer, model) -> np.ndarray:
    """
    Encode a list of strings with FinBERT.
    Empty strings → zero vector (no-news days).
    Returns float32 array of shape (N, 768).
    """
    n = len(headlines)
    embeddings = np.zeros((n, model.config.hidden_size), dtype=np.float32)

    # Indices with actual content
    non_empty = [(i, h) for i, h in enumerate(headlines) if h.strip()]

    for batch_start in tqdm(range(0, len(non_empty), BATCH_SIZE), desc="Encoding batches"):
        batch = non_empty[batch_start : batch_start + BATCH_SIZE]
        idxs, texts = zip(*batch)

        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        # Use CLS token (index 0) — standard for FinBERT classification tasks
        cls = output.last_hidden_state[:, 0, :].float().cpu().numpy()
        for i, idx in enumerate(idxs):
            embeddings[idx] = cls[i]

    return embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n=== Loading features ===")
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    headlines = df["headline"].fillna("").tolist()
    n_empty = sum(1 for h in headlines if not h.strip())
    print(f"  Rows: {len(headlines)}  |  No-news (zero vector): {n_empty} ({n_empty/len(headlines):.1%})")

    print("\n=== Loading FinBERT ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    print(f"  Model loaded → {DEVICE}")

    print("\n=== Encoding headlines ===")
    embeddings = encode_headlines(headlines, tokenizer, model)
    out_path = PROCESSED_DIR / "embeddings.npy"
    np.save(out_path, embeddings)
    print(f"  Saved {embeddings.shape} to {out_path}")

    print("\n=== Fitting PCA ({PCA_DIM}-d) ===")
    # Fit only on non-zero rows (i.e. rows with actual news)
    non_zero_mask = embeddings.any(axis=1)
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(embeddings[non_zero_mask])
    print(f"  Variance explained by {PCA_DIM} components: "
          f"{pca.explained_variance_ratio_.sum():.1%}")

    # Transform all rows (zero rows → near-zero in PCA space, which is fine)
    embeddings_pca = pca.transform(embeddings).astype(np.float32)
    np.save(PROCESSED_DIR / "embeddings_pca.npy", embeddings_pca)
    with open(PROCESSED_DIR / "pca.pkl", "wb") as f:
        pickle.dump(pca, f)
    print(f"  Saved embeddings_pca.npy ({embeddings_pca.shape}) and pca.pkl")

    print("\nDone. Next step: python src/train.py")


if __name__ == "__main__":
    main()
