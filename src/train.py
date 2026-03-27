"""
train.py
--------
Full training pipeline for FinBERT-Attention-LSTM.

Models
------
  logreg              — Logistic Regression (required linear baseline)
  rf_notxt            — Random Forest, price only
  rf_full             — Random Forest, price + FinBERT text
  lstm                — Plain LSTM, price only
  selfattlstm_notxt   — Self-Attention LSTM, price only (ablation)
  selfattlstm_full    — Self-Attention LSTM, price + text (main model)

Steps
-----
  1. Classical baselines: LR (no tuning), RF notxt + RF full (Optuna if --tune).
  2. Neural models: Optuna search (if --tune) → 3-seed final training.
  3. Saves:
       results/metrics.json                      — per-ticker and aggregated mean ± std
       results/predictions/<model>_<ticker>.npz  — raw probs/labels/dates/tickers

Usage
-----
  python src/train.py                        # default params, 3 seeds
  python src/train.py --tune                 # Optuna tuning, then 3 seeds
  python src/train.py --tune --trials 50 --seeds 3
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from models import (
    LSTMModel,
    LogisticRegressionModel,
    SelfAttLSTM,
    N_PRICE_FEATURES,
    N_FULL_FEATURES,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # Returns & momentum
    "return_1d", "return_5d", "return_20d",
    # Volatility
    "vol_20d",
    # Trend
    "ma_ratio",
    # Oscillators
    "rsi", "macd_signal",
    # Volume
    "vol_change", "turnover_ratio",
    # OHLCV candlestick
    "intraday_ret", "hl_range", "upper_shadow", "lower_shadow",
]

WINDOW        = 15
N_SEEDS       = 3
N_TUNE_TRIALS = 50
MAX_EPOCHS    = 100
PATIENCE      = 10

TRAIN_END = pd.Timestamp("2017-12-31").date()
VAL_END   = pd.Timestamp("2018-12-31").date()

TICKERS = ["MRK", "JNJ", "KO", "NVDA", "QCOM"]

PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR   = ROOT / "results"
PRED_DIR      = RESULTS_DIR / "predictions"
OPTUNA_DB     = ROOT / "optuna_studies.db"

# ---------------------------------------------------------------------------
# Device & reproducibility
# ---------------------------------------------------------------------------
def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = _get_device()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class StockDataset(Dataset):
    """
    Sliding-window dataset for a single split.

    Stores per-sample metadata (date, ticker) in self.meta so that raw
    predictions can be saved with their timestamps for later analysis.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        scaler: MinMaxScaler,
        use_text: bool = True,
        window: int = WINDOW,
    ) -> None:
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.meta: list[tuple[str, str]] = []   # (date_str, ticker)

        for ticker, grp in df.groupby("Ticker"):
            grp      = grp.sort_values("date")
            orig_idx = grp.index.values
            price    = scaler.transform(grp[FEATURE_COLS].values).astype(np.float32)
            embs     = embeddings[orig_idx].astype(np.float32)
            labels   = grp["target"].values.astype(np.float32)
            dates    = grp["date"].values

            for i in range(window - 1, len(grp)):
                sl   = slice(i - window + 1, i + 1)
                feat = np.concatenate([price[sl], embs[sl]], axis=1) if use_text else price[sl]
                self.samples.append((
                    torch.from_numpy(feat),
                    torch.tensor(labels[i]).unsqueeze(0),
                ))
                self.meta.append((str(dates[i]), ticker))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def make_loader(ds: StockDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def flatten_windows(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    scaler: MinMaxScaler,
    use_text: bool = False,
    window: int = WINDOW,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """
    Flatten sliding windows into a 2-D feature matrix for classical models.
    Returns (X, y, meta) where meta is list of (date_str, ticker).
    Price-only:  X shape (N, window * 13)  =  (N, 195)
    Full:        X shape (N, window * 77)  =  (N, 1155)
    """
    X_list, y_list, meta = [], [], []
    for ticker, grp in df.groupby("Ticker"):
        grp      = grp.sort_values("date")
        orig_idx = grp.index.values
        price    = scaler.transform(grp[FEATURE_COLS].values).astype(np.float32)
        embs     = embeddings[orig_idx].astype(np.float32)
        labels   = grp["target"].values
        dates    = grp["date"].values

        for i in range(window - 1, len(grp)):
            sl = slice(i - window + 1, i + 1)
            p  = price[sl].flatten()
            row = np.concatenate([p, embs[sl].flatten()]) if use_text else p
            X_list.append(row)
            y_list.append(labels[i])
            meta.append((str(dates[i]), ticker))

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        meta,
    )

# ---------------------------------------------------------------------------
# Raw-prediction saving
# ---------------------------------------------------------------------------
def save_predictions(
    model_name: str,
    split_data: dict[str, tuple[np.ndarray, np.ndarray, list]],
) -> None:
    """
    Save raw probabilities, labels, dates, tickers for all splits.

    split_data: {"train": (probs, labels, meta), "val": ..., "test": ...}
    Output: results/predictions/<model_name>.npz
    """
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    arrays: dict = {}
    for split, (probs, labels, meta) in split_data.items():
        dates   = np.array([m[0] for m in meta])
        tickers = np.array([m[1] for m in meta])
        arrays[f"{split}_probs"]   = probs.astype(np.float32)
        arrays[f"{split}_labels"]  = labels.astype(np.float32)
        arrays[f"{split}_dates"]   = dates
        arrays[f"{split}_tickers"] = tickers
    np.savez(PRED_DIR / f"{model_name}.npz", **arrays)
    print(f"  Predictions → {PRED_DIR / model_name}.npz")

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_raw(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, labels) arrays for a loader."""
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device)).cpu())
        all_labels.append(y)
    logits = torch.cat(all_logits).squeeze(1)
    labels = torch.cat(all_labels).squeeze(1).numpy()
    probs  = torch.sigmoid(logits).numpy()
    return probs, labels


def _metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = (probs >= 0.5).astype(int)
    auc   = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    return {
        "acc": float(accuracy_score(labels, preds)),
        "f1":  float(f1_score(labels, preds, average="macro", zero_division=0)),
        "auc": float(auc),
    }


@torch.no_grad()
def evaluate(model, loader, device, pos_weight=None):
    """Returns (loss, acc, f1, auc)."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device) if pos_weight is not None else None
    )
    all_logits, all_labels = [], []
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(x)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits).squeeze(1)
    labels = torch.cat(all_labels).squeeze(1).numpy()
    probs  = torch.sigmoid(logits).numpy()
    m = _metrics(probs, labels)
    return total_loss / len(loader.dataset), m["acc"], m["f1"], m["auc"]


def train_neural(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    weight_decay: float,
    pos_weight: torch.Tensor,
    grad_clip: float = 1.0,
    device: torch.device = DEVICE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    trial: "optuna.trial.Trial | None" = None,
) -> dict:
    """Train with early stopping on val AUC. Returns best val metrics."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    best_auc   = 0.0
    best_state = None
    wait       = 0

    for epoch in range(max_epochs):
        _train_epoch(model, train_loader, optimizer, criterion, device, grad_clip)
        _, val_acc, val_f1, val_auc = evaluate(model, val_loader, device, pos_weight)

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1

        if trial is not None:
            trial.report(val_auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, val_acc, val_f1, val_auc = evaluate(model, val_loader, device, pos_weight)
    return {"val_acc": val_acc, "val_f1": val_f1, "val_auc": val_auc}

# ---------------------------------------------------------------------------
# Optuna objectives — expanded search space
# ---------------------------------------------------------------------------
def _selfattlstm_objective(trial, train_ds, val_ds, input_dim, pos_weight):
    hidden_dim   = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    # All of [1,2,4,8] divide all of [32,64,128,256] — static list avoids
    # CategoricalDistribution dynamic value space error across resumed studies
    n_heads      = trial.suggest_categorical("n_heads", [1, 2, 4, 8])
    num_layers   = trial.suggest_categorical("num_layers", [1, 2])
    attn_dropout = trial.suggest_float("attn_dropout", 0.0, 0.3)
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.1, 0.5)
    lr           = trial.suggest_float("lr", 5e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    grad_clip    = trial.suggest_categorical("grad_clip", [0.5, 1.0, 2.0, 5.0])

    set_seed(42)
    model = SelfAttLSTM(input_dim=input_dim, hidden_dim=hidden_dim,
                        n_heads=n_heads, dropout=lstm_dropout)
    res = train_neural(
        model,
        make_loader(train_ds, batch_size, shuffle=True),
        make_loader(val_ds,   batch_size, shuffle=False),
        lr=lr, weight_decay=weight_decay,
        pos_weight=pos_weight, grad_clip=grad_clip,
        trial=trial,
    )
    return res["val_auc"]


def _lstm_objective(trial, train_ds, val_ds, input_dim, pos_weight):
    hidden_dim   = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    num_layers   = trial.suggest_categorical("num_layers", [1, 2])
    dropout      = trial.suggest_float("dropout", 0.1, 0.5)
    lr           = trial.suggest_float("lr", 5e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    grad_clip    = trial.suggest_categorical("grad_clip", [0.5, 1.0, 2.0, 5.0])

    set_seed(42)
    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim,
                      num_layers=num_layers, dropout=dropout)
    res = train_neural(
        model,
        make_loader(train_ds, batch_size, shuffle=True),
        make_loader(val_ds,   batch_size, shuffle=False),
        lr=lr, weight_decay=weight_decay,
        pos_weight=pos_weight, grad_clip=grad_clip,
        trial=trial,
    )
    return res["val_auc"]


def _rf_objective(trial, X_tr, y_tr, X_va, y_va):
    n_estimators     = trial.suggest_categorical("n_estimators", [100, 300, 500])
    max_depth        = trial.suggest_categorical("max_depth", [None, 10, 20, 30])
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_va)[:, 1]
    return float(roc_auc_score(y_va, probs))


def tune(
    study_name: str,
    objective,
    n_trials: int = N_TUNE_TRIALS,
) -> dict:
    """
    Run or resume an Optuna study. Returns best hyperparameter dict.
    Studies persist in OPTUNA_DB — interrupted runs resume automatically.
    """
    storage = f"sqlite:///{OPTUNA_DB}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    done      = sum(1 for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = max(0, n_trials - done)
    print(f"\n[Optuna] '{study_name}' — {done} done, {remaining} remaining "
          f"(target {n_trials})  db: {OPTUNA_DB.name}")

    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)

    print(f"  Best val_auc : {study.best_value:.4f}")
    print(f"  Best params  : {study.best_params}")
    return study.best_params


def get_best_params_if_exists(study_name: str) -> dict | None:
    """
    Load best params from a previously completed Optuna study, if one exists.
    Returns None if no study or no completed trials are found.
    Used to automatically apply prior tuning results without re-running --tune.
    """
    if not OPTUNA_DB.exists():
        return None
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{OPTUNA_DB}",
        )
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            return None
        print(f"  [Optuna] Loaded {len(completed)} prior trials for '{study_name}' "
              f"— best val_auc: {study.best_value:.4f}")
        return study.best_params
    except Exception:
        return None


def resolve_params(model_name: str, study_name: str, models_to_tune: set,
                   objective, n_trials: int) -> dict:
    """
    Decide which hyperparameters to use for a model:
      1. model in models_to_tune  → run / resume Optuna study (keyed by study_name)
      2. prior study exists       → load best params silently
      3. fallback                 → use _DEFAULT_PARAMS
    Always merges with defaults so no key is missing.
    """
    defaults = _DEFAULT_PARAMS[model_name]
    if model_name in models_to_tune:
        best = tune(study_name, objective, n_trials=n_trials)
    else:
        best = get_best_params_if_exists(study_name)
        if best is None:
            print(f"  No prior study found for '{study_name}' — using defaults.")
    return {**defaults, **best} if best else defaults


# ---------------------------------------------------------------------------
# Default hyperparameters (used when --tune is not set)
# ---------------------------------------------------------------------------
_DEFAULT_PARAMS: dict[str, dict] = {
    "lstm": {
        "hidden_dim": 64, "num_layers": 1, "dropout": 0.2,
        "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 32, "grad_clip": 1.0,
    },
    "selfattlstm_notxt": {
        "hidden_dim": 64, "n_heads": 4, "num_layers": 1,
        "attn_dropout": 0.1, "lstm_dropout": 0.2,
        "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 32, "grad_clip": 1.0,
    },
    "selfattlstm_full": {
        "hidden_dim": 64, "n_heads": 4, "num_layers": 1,
        "attn_dropout": 0.1, "lstm_dropout": 0.2,
        "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 32, "grad_clip": 1.0,
    },
    "rf_notxt": {
        "n_estimators": 300, "max_depth": None,
        "min_samples_leaf": 1, "max_features": "sqrt",
    },
    "rf_full": {
        "n_estimators": 300, "max_depth": None,
        "min_samples_leaf": 1, "max_features": "sqrt",
    },
}

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def _build_neural(model_name: str, params: dict) -> nn.Module:
    h = params.get("hidden_dim", 64)
    if model_name == "lstm":
        return LSTMModel(
            input_dim=N_PRICE_FEATURES,
            hidden_dim=h,
            num_layers=params.get("num_layers", 1),
            dropout=params.get("dropout", 0.2),
        )
    if model_name == "selfattlstm_notxt":
        return SelfAttLSTM(
            input_dim=N_PRICE_FEATURES,
            hidden_dim=h,
            n_heads=params.get("n_heads", 4),
            dropout=params.get("lstm_dropout", params.get("dropout", 0.2)),
        )
    if model_name == "selfattlstm_full":
        return SelfAttLSTM(
            input_dim=N_FULL_FEATURES,
            hidden_dim=h,
            n_heads=params.get("n_heads", 4),
            dropout=params.get("lstm_dropout", params.get("dropout", 0.2)),
        )
    raise ValueError(f"Unknown neural model: {model_name}")


def _build_rf(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", None),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_features=params.get("max_features", "sqrt"),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

# ---------------------------------------------------------------------------
# Classical baselines (LR, RF notxt, RF full)
# ---------------------------------------------------------------------------
def run_classical(
    ticker: str,
    train_df, val_df, test_df,
    embeddings, scaler,
    models_to_tune: set,
    n_trials: int = N_TUNE_TRIALS,
    n_seeds: int = N_SEEDS,
) -> dict:
    """
    Runs Logistic Regression, RF (no text), RF (full) for a single ticker.
    Returns results dict and saves raw predictions keyed by model_ticker.
    """
    results = {}

    # Flatten windows for price-only and full variants
    X_tr_p, y_tr, meta_tr_p = flatten_windows(train_df, embeddings, scaler, use_text=False)
    X_va_p, y_va, meta_va_p = flatten_windows(val_df,   embeddings, scaler, use_text=False)
    X_te_p, y_te, meta_te_p = flatten_windows(test_df,  embeddings, scaler, use_text=False)

    X_tr_f, _,    meta_tr_f = flatten_windows(train_df, embeddings, scaler, use_text=True)
    X_va_f, _,    meta_va_f = flatten_windows(val_df,   embeddings, scaler, use_text=True)
    X_te_f, _,    meta_te_f = flatten_windows(test_df,  embeddings, scaler, use_text=True)

    # ── Logistic Regression (multi-seed for consistent reporting) ────────────
    print("\n=== Logistic Regression ===")
    split_keys   = ["train", "val", "test"]
    split_data_p = [
        (X_tr_p, y_tr, meta_tr_p),
        (X_va_p, y_va, meta_va_p),
        (X_te_p, y_te, meta_te_p),
    ]
    lr_seed_metrics = {f"{s}_{k}": [] for s in split_keys for k in ("acc", "f1", "auc")}
    lr_best_val_auc = -1.0
    lr_best_splits  = {}

    for seed in range(n_seeds):
        lr_clf = LogisticRegressionModel(random_state=seed)
        lr_clf.fit(X_tr_p, y_tr)

        seed_splits = {}
        for sname, (X_s, y_s, meta_s) in zip(split_keys, split_data_p):
            probs = lr_clf.predict_proba(X_s)
            seed_splits[sname] = (probs, y_s, meta_s)
            m = _metrics(probs, y_s)
            for k, v in m.items():
                lr_seed_metrics[f"{sname}_{k}"].append(v)

        val_auc = lr_seed_metrics["val_auc"][-1]
        print(f"  seed={seed}  train_auc={lr_seed_metrics['train_auc'][-1]:.4f}"
              f"  val_auc={val_auc:.4f}  test_auc={lr_seed_metrics['test_auc'][-1]:.4f}")
        if val_auc > lr_best_val_auc:
            lr_best_val_auc = val_auc
            lr_best_splits  = seed_splits

    save_predictions(f"logreg_{ticker}", lr_best_splits)
    results["logreg"] = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in lr_seed_metrics.items()
    }

    # ── Random Forest helpers ───────────────────────────────────────────────
    def _run_rf(name, X_tr, X_va, X_te, y_tr, y_va, y_te,
                meta_tr, meta_va, meta_te):
        study_name = f"{name}_{ticker}"
        obj        = lambda t: _rf_objective(t, X_tr, y_tr, X_va, y_va)
        params     = resolve_params(name, study_name, models_to_tune, obj, n_trials)

        print(f"\n=== {name} ===")
        print(f"  Params: {params}")

        # Run n_seeds with different random_state values for mean ± std
        split_keys   = ["train", "val", "test"]
        split_data   = [
            (X_tr, y_tr, meta_tr),
            (X_va, y_va, meta_va),
            (X_te, y_te, meta_te),
        ]
        seed_metrics = {f"{s}_{k}": [] for s in split_keys for k in ("acc", "f1", "auc")}
        best_val_auc = -1.0
        best_splits  = {}

        for seed in range(n_seeds):
            clf = RandomForestClassifier(
                **{k: v for k, v in params.items()},
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
            clf.fit(X_tr, y_tr)

            seed_splits = {}
            for sname, (X_s, y_s, meta_s) in zip(split_keys, split_data):
                probs = clf.predict_proba(X_s)[:, 1]
                seed_splits[sname] = (probs, y_s, meta_s)
                m = _metrics(probs, y_s)
                for k, v in m.items():
                    seed_metrics[f"{sname}_{k}"].append(v)

            val_auc = seed_metrics["val_auc"][-1]
            print(f"  seed={seed}  train_auc={seed_metrics['train_auc'][-1]:.4f}"
                  f"  val_auc={val_auc:.4f}  test_auc={seed_metrics['test_auc'][-1]:.4f}")
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_splits  = seed_splits

        save_predictions(f"{name}_{ticker}", best_splits)
        summary = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in seed_metrics.items()
        }
        return summary, params

    rf_notxt_res, rf_notxt_params = _run_rf(
        "rf_notxt",
        X_tr_p, X_va_p, X_te_p, y_tr, y_va, y_te,
        meta_tr_p, meta_va_p, meta_te_p,
    )
    results["rf_notxt"] = rf_notxt_res

    rf_full_res, rf_full_params = _run_rf(
        "rf_full",
        X_tr_f, X_va_f, X_te_f, y_tr, y_va, y_te,
        meta_tr_f, meta_va_f, meta_te_f,
    )
    results["rf_full"] = rf_full_res

    return results



# ---------------------------------------------------------------------------
# Neural: multi-seed training
# ---------------------------------------------------------------------------
def run_seeds(
    model_name: str,
    file_key: str,
    params: dict,
    train_ds: StockDataset,
    val_ds: StockDataset,
    test_ds: StockDataset,
    pos_weight: torch.Tensor,
    n_seeds: int = N_SEEDS,
) -> dict:
    """
    Train model across n_seeds seeds. Saves raw predictions from best seed.
    Returns summary dict with mean±std across seeds.
    file_key is used for prediction file naming (e.g. 'lstm_MRK').
    """
    bs           = params.get("batch_size",   32)
    lr           = params.get("lr",           1e-3)
    weight_decay = params.get("weight_decay", 1e-5)
    grad_clip    = params.get("grad_clip",    1.0)

    train_loader = make_loader(train_ds, bs, shuffle=True)
    val_loader   = make_loader(val_ds,   bs, shuffle=False)
    test_loader  = make_loader(test_ds,  bs, shuffle=False)

    train_runs, val_runs, test_runs = [], [], []
    best_val_auc = -1.0
    best_model   = None

    for seed in range(n_seeds):
        set_seed(seed)
        model = _build_neural(model_name, params)
        vres  = train_neural(model, train_loader, val_loader,
                             lr=lr, weight_decay=weight_decay,
                             pos_weight=pos_weight, grad_clip=grad_clip)
        val_runs.append(vres)

        _, t_acc, t_f1, t_auc = evaluate(model, test_loader, DEVICE, pos_weight)
        _, tr_acc, tr_f1, tr_auc = evaluate(model, train_loader, DEVICE, pos_weight)
        test_runs.append({"test_acc": t_acc, "test_f1": t_f1, "test_auc": t_auc})
        train_runs.append({"train_acc": tr_acc, "train_f1": tr_f1, "train_auc": tr_auc})

        print(f"  seed={seed}  train_auc={tr_auc:.4f}  "
              f"val_auc={vres['val_auc']:.4f}  test_auc={t_auc:.4f}")

        if vres["val_auc"] > best_val_auc:
            best_val_auc = vres["val_auc"]
            best_model   = model

    # Save raw predictions from best seed's model
    if best_model is not None:
        splits_pred = {}
        for sname, loader, ds in [
            ("train", train_loader, train_ds),
            ("val",   val_loader,   val_ds),
            ("test",  test_loader,  test_ds),
        ]:
            probs, labels = predict_raw(best_model, loader, DEVICE)
            splits_pred[sname] = (probs, labels, ds.meta)
        save_predictions(file_key, splits_pred)

    def _stat(runs, key):
        vals = [r[key] for r in runs]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    summary = {
        "train_acc": _stat(train_runs, "train_acc"),
        "train_f1":  _stat(train_runs, "train_f1"),
        "train_auc": _stat(train_runs, "train_auc"),
        "val_acc":   _stat(val_runs,   "val_acc"),
        "val_f1":    _stat(val_runs,   "val_f1"),
        "val_auc":   _stat(val_runs,   "val_auc"),
        "test_acc":  _stat(test_runs,  "test_acc"),
        "test_f1":   _stat(test_runs,  "test_f1"),
        "test_auc":  _stat(test_runs,  "test_auc"),
        "best_params": params,
    }
    return summary

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    TUNABLE = {"rf_notxt", "rf_full", "lstm", "selfattlstm_notxt", "selfattlstm_full"}

    parser = argparse.ArgumentParser(
        description="FinBERT-Attention-LSTM training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Tuning examples:\n"
            "  --tune                         tune all models\n"
            "  --tune lstm selfattlstm_full   tune specific models only\n"
            "  (no --tune)                    use prior Optuna results if available,\n"
            "                                 else fall back to defaults\n"
        ),
    )
    parser.add_argument(
        "--tune", nargs="*", metavar="MODEL",
        help=f"Models to tune. Omit for all, or list from: {', '.join(sorted(TUNABLE))}",
    )
    parser.add_argument("--trials", type=int, default=N_TUNE_TRIALS)
    parser.add_argument("--seeds",  type=int, default=N_SEEDS)
    args = parser.parse_args()

    # --tune        → tune all models
    # --tune a b    → tune only a, b
    # (omitted)     → no tuning (use prior study or defaults)
    if args.tune is None:
        models_to_tune: set = set()
    elif len(args.tune) == 0:
        models_to_tune = set(TUNABLE)
    else:
        invalid = set(args.tune) - TUNABLE
        if invalid:
            parser.error(f"Unknown model(s) for --tune: {invalid}. "
                         f"Valid: {', '.join(sorted(TUNABLE))}")
        models_to_tune = set(args.tune)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n=== Loading data ===")
    full_df    = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    full_df["date"] = pd.to_datetime(full_df["Date"]).dt.date
    embeddings = np.load(PROCESSED_DIR / "embeddings_pca.npy")

    train_df = full_df[full_df["date"] <= TRAIN_END].copy()
    val_df   = full_df[(full_df["date"] > TRAIN_END) & (full_df["date"] <= VAL_END)].copy()
    test_df  = full_df[full_df["date"] > VAL_END].copy()
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"  Device: {DEVICE}")

    # ── Per-ticker training loop ──────────────────────────────────────────────
    ticker_results: dict[str, dict] = {}

    for ticker in TICKERS:
        print(f"\n{'=' * 60}")
        print(f"TICKER: {ticker}")
        print(f"{'=' * 60}")

        tr = train_df[train_df["Ticker"] == ticker].copy()
        va = val_df[val_df["Ticker"] == ticker].copy()
        te = test_df[test_df["Ticker"] == ticker].copy()

        if len(tr) < WINDOW or len(va) < WINDOW or len(te) < WINDOW:
            print(f"  WARNING: {ticker} has insufficient rows — skipping.")
            continue

        # Per-ticker scaler (fit on this ticker's training data only)
        scaler = MinMaxScaler()
        scaler.fit(tr[FEATURE_COLS].values)

        # Per-ticker class weight
        y_tr_all = tr["target"].values
        n_neg    = (y_tr_all == 0).sum()
        n_pos    = (y_tr_all == 1).sum()
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
        print(f"  Class balance — pos: {n_pos} ({n_pos/(n_pos+n_neg):.1%})  "
              f"neg: {n_neg}  pos_weight: {pos_weight.item():.3f}")

        ticker_results[ticker] = {}

        # ── Classical baselines ───────────────────────────────────────────
        classical_res = run_classical(
            ticker, tr, va, te, embeddings, scaler,
            models_to_tune=models_to_tune, n_trials=args.trials, n_seeds=args.seeds,
        )
        ticker_results[ticker].update(classical_res)

        # ── Neural datasets ───────────────────────────────────────────────
        print(f"\nBuilding sliding-window datasets for {ticker} ...")
        kw = dict(embeddings=embeddings, scaler=scaler, window=WINDOW)
        train_price = StockDataset(tr, use_text=False, **kw)
        val_price   = StockDataset(va, use_text=False, **kw)
        test_price  = StockDataset(te, use_text=False, **kw)
        train_full  = StockDataset(tr, use_text=True,  **kw)
        val_full    = StockDataset(va, use_text=True,  **kw)
        test_full   = StockDataset(te, use_text=True,  **kw)
        print(f"  Price-only: train={len(train_price)} val={len(val_price)} test={len(test_price)}")
        print(f"  Full:       train={len(train_full)}  val={len(val_full)}  test={len(test_full)}")

        # ── Neural models ─────────────────────────────────────────────────
        neural_configs = [
            ("lstm",              N_PRICE_FEATURES, train_price, val_price, test_price),
            ("selfattlstm_notxt", N_PRICE_FEATURES, train_price, val_price, test_price),
            ("selfattlstm_full",  N_FULL_FEATURES,  train_full,  val_full,  test_full),
        ]

        for model_name, input_dim, tr_ds, va_ds, te_ds in neural_configs:
            file_key   = f"{model_name}_{ticker}"
            study_name = file_key
            print(f"\n=== {file_key} ===")

            if model_name == "lstm":
                obj = lambda t, d=input_dim: _lstm_objective(t, tr_ds, va_ds, d, pos_weight)
            else:
                obj = lambda t, d=input_dim: _selfattlstm_objective(t, tr_ds, va_ds, d, pos_weight)

            params = resolve_params(model_name, study_name, models_to_tune, obj, args.trials)
            print(f"  Params: {params}")

            summary = run_seeds(
                model_name, file_key, params, tr_ds, va_ds, te_ds,
                pos_weight=pos_weight, n_seeds=args.seeds,
            )
            ticker_results[ticker][model_name] = summary
            print(f"  train_auc={summary['train_auc']['mean']:.4f}  "
                  f"val_auc={summary['val_auc']['mean']:.4f}  "
                  f"test_auc={summary['test_auc']['mean']:.4f}")

    # ── Aggregate across tickers ──────────────────────────────────────────────
    all_models = ["logreg", "rf_notxt", "rf_full", "lstm", "selfattlstm_notxt", "selfattlstm_full"]
    metric_keys = ["train_acc", "train_f1", "train_auc", "val_acc", "val_f1", "val_auc",
                   "test_acc", "test_f1", "test_auc"]

    aggregated: dict[str, dict] = {}
    for model in all_models:
        lists: dict[str, list] = {k: [] for k in metric_keys}
        for ticker, res in ticker_results.items():
            if model not in res:
                continue
            for k in metric_keys:
                if k in res[model]:
                    lists[k].append(res[model][k]["mean"])
        if any(lists.values()):
            aggregated[model] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in lists.items() if v
            }

    # ── Save & print ──────────────────────────────────────────────────────────
    out = {"per_ticker": ticker_results, "aggregated": aggregated}
    out_path = RESULTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults → {out_path}")

    print("\n" + "=" * 84)
    print(f"{'Model':<25} {'Train AUC':>12} {'Val AUC':>12} {'Test AUC':>12} {'Test F1':>12}")
    print(f"  (mean ± std across {len(ticker_results)} tickers)")
    print("-" * 84)
    for name in all_models:
        if name not in aggregated:
            continue
        res = aggregated[name]
        def _f(k):
            return f"{res[k]['mean']:.4f}±{res[k]['std']:.3f}" if k in res else "    —    "
        tr = _f("train_auc")
        print(f"{name:<25} {tr:>12} {_f('val_auc'):>12} {_f('test_auc'):>12} {_f('test_f1'):>12}")
    print("=" * 84)


if __name__ == "__main__":
    main()
