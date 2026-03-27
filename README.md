# Self-Att-LSTM for Stock Direction Prediction

Per-stock binary classification of next-day return direction (up/down) for individual S&P 500 constituent stocks, using a Self-Attention LSTM fusing FinBERT news embeddings with price features. Extends Sun et al. (2026) to English news + US equities + classification.

## Setup

```bash
conda env create -f environment.yml
```

> Place Kaggle Benzinga files at `data/raw/news/` before running. Required file: `analyst_ratings_processed.csv`.

## Pipeline

Use `main.py` as the single entry point — it handles step ordering, skips steps whose outputs already exist, and passes flags through to the relevant scripts.

```bash
# Run everything end-to-end (first time):
python main.py --all

# Training only (data + embeddings already on disk):
python main.py --steps train

# Tune all models then train (30 trials per model per ticker):
python main.py --steps train --tune --trials 30 --seeds 3

# No --tune: uses prior Optuna results if available, else defaults:
python main.py --steps train

# Full run, tune all:
python main.py --all --tune

# Force re-run a step even if its output exists:
python main.py --steps embed --force
```

> **Note:** Delete `optuna_studies.db` before the first tuned run to avoid stale pooled-model studies from a prior pipeline version.

### Steps

| Step | Script | Output | Notes |
|---|---|---|---|
| `pipeline` | `src/data_pipeline.py` | `data/processed/features.parquet` | Run once; skipped if output exists |
| `embed` | `src/embed_news.py` | `data/processed/embeddings_pca.npy` | Slow (~1 hr CPU); skipped if output exists |
| `train` | `src/train.py` | `results/metrics.json` | Trains one model per ticker; add `--tune` for Optuna search |

Individual scripts can still be run directly if preferred:

```bash
python src/data_pipeline.py
python src/embed_news.py
python src/train.py [--tune] [--trials 50] [--seeds 3]
```

### Per-ticker training approach

The training pipeline trains **one independent model per ticker per model-type** (e.g. `lstm_MRK`, `lstm_JNJ`, …). Each ticker receives:
- Its own `MinMaxScaler` fit only on that ticker's training rows
- Its own `pos_weight` computed from that ticker's class balance
- Its own Optuna study (e.g. `selfattlstm_full_NVDA`) so hyperparameter trials do not cross-contaminate across tickers

Outputs are named `<model>_<ticker>`:
- Predictions: `results/predictions/lstm_MRK.npz`, `logreg_JNJ.npz`, …
- `results/metrics.json` contains both `per_ticker` (raw per-model-ticker results) and `aggregated` (mean ± std across all 5 tickers) sections

Runtime (observed on Apple M-series): ~1.2 min without tuning, ~29 min with 30 Optuna trials per model per ticker (default via `main.py`).

## Models

| # | Model | Features | Input dim | Role |
|---|---|---|---|---|
| B0 | Logistic Regression | Price (flat window) | 195 | Required linear baseline |
| B1 | Random Forest (no text) | Price (flat window) | 195 | Non-linear classical baseline |
| B2 | Random Forest (full) | Price + FinBERT (flat window) | 1,155 | Classical baseline with text |
| A1 | Plain LSTM | Price sequence | 15 × 13 | Neural temporal baseline |
| A2 | Self-Att-LSTM (no text) | Price sequence | 15 × 13 | Ablation: attention without text |
| M | Self-Att-LSTM (full) | Price + FinBERT sequence | 15 × 77 | Main model |

The model progression isolates each design choice:
- **LR → RF (no text):** does non-linearity help?
- **RF no text → RF full:** does text help in classical ML?
- **RF → LSTM:** does temporal modelling help?
- **LSTM → Self-Att-LSTM (no text):** does attention help?
- **Self-Att-LSTM no text → full:** does text help with deep learning?

### B0. Logistic Regression

```
Input:  (T × 13) price features → flattened → (195,)
Linear: 195 → 1  +  sigmoid → P(up)
```

`class_weight='balanced'`, L2 regularisation, `lbfgs` solver.

### B1 / B2. Random Forest

```
Input:  price only  → flattened → (195,)      [B1]
        price + text → flattened → (1,155,)   [B2: 15×13 + 15×64]
500 trees, class_weight='balanced'
Hyperparameters tuned via Optuna (n_estimators, max_depth,
min_samples_leaf, max_features)
```

### A1. Plain LSTM

```
Input:  (T=15, 13)
LSTM:   input=13, hidden=64, layers=1–2
Dropout → Linear(64→1) → sigmoid → P(up)
```

### A2 / M. Self-Att-LSTM

```
Input:  (T=15, d_in)
          d_in = 13  [no text]
          d_in = 77  [full: 13 price + 64 FinBERT-PCA]

Linear projection:  d_in → hidden_dim

Multi-Head Self-Attention (Q=K=V):
  heads ∈ {1,2,4,8},  attn_dropout ∈ [0.0, 0.3]

Residual + LayerNorm

LSTM:   hidden_dim, layers=1–2,  lstm_dropout ∈ [0.1, 0.5]

Linear(hidden_dim → 1) → sigmoid → P(up)
```

Attention weights `(batch, T, T)` stored after each forward pass for visualisation.

Loss: `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)` — corrects for class imbalance.

## Dataset

### Price data
- **Source:** Yahoo Finance via `yfinance`
- **Tickers:** MRK, JNJ, KO, NVDA, QCOM (5 S&P 500 constituents, diverse sectors)
- **Period:** 2013-01-01 – 2020-06-30 (7.5 years, 1,886 trading days per ticker)
- **Frequency:** Daily OHLCV

**Price features (13 total):**

| Feature | Group | Description |
|---|---|---|
| `return_1d` | Returns | Daily close-to-close return |
| `return_5d` | Returns | 5-day cumulative return |
| `return_20d` | Returns | 20-day cumulative return |
| `vol_20d` | Volatility | 20-day rolling std of returns |
| `ma_ratio` | Trend | 5-day MA / 20-day MA |
| `rsi` | Oscillator | 14-day RSI |
| `macd_signal` | Oscillator | MACD signal line (EMA 9 of MACD) |
| `vol_change` | Volume | 1-day volume change |
| `turnover_ratio` | Volume | Volume / 20-day avg volume |
| `intraday_ret` | Candlestick | (Close − Open) / Open |
| `hl_range` | Candlestick | (High − Low) / Close |
| `upper_shadow` | Candlestick | Upper wick / (High − Low) |
| `lower_shadow` | Candlestick | Lower wick / (High − Low) |

**Target:** `y = 1` if next-day close > today's close, else `y = 0`. Class balance: ~52% up / 48% down.

### News data
- **Source:** Kaggle Benzinga (`analyst_ratings_processed.csv`)
- **Coverage:** 2009–Jun 2020, minute-precision timestamps (UTC-4)
- **Rows after filtering to 5 tickers + experiment window:** 5,356 (date, ticker) pairs
- **No-news days:** 47.3% of rows — represented as zero vectors

**News features:**
- Headlines aggregated per (date, ticker) into one document
- Encoded with **FinBERT** (`ProsusAI/finbert`, 836 MB, cached locally)
- CLS token embedding: 768-d → **PCA → 64-d** (85.8% variance explained)
- Leakage guard: news from day `t−1` predicts return on day `t`

### Splits

| Split | Period | Rows |
|---|---|---|
| Train | 2013-01-31 – 2017-12-29 | 6,195 |
| Val | 2018-01-02 – 2018-12-31 | 1,255 |
| Test | 2019-01-02 – 2020-06-26 | 1,875 |

Splits are strictly temporal — no shuffling at any stage.

## Key settings

| Parameter | Value |
|---|---|
| Tickers | MRK, JNJ, KO, NVDA, QCOM |
| Period | 2013–Jun 2020 |
| News source | Kaggle Benzinga `analyst_ratings_processed.csv` |
| Window | T = 15 trading days |
| Split | Train 2013–2017 / Val 2018 / Test 2019–Jun 2020 |
| Training approach | Per-ticker — one independent model per ticker per model-type |
| Scaler | MinMaxScaler fit on each ticker's training rows only |
| Seeds | 3 runs per model-ticker (random_state for classical, torch seed for neural) — mean ± std reported |
| Hyperparameter tuning | Optuna TPE sampler, MedianPruner, 30 trials per model per ticker (default via `main.py --tune`); study name = `<model>_<ticker>` |
| Loss | BCEWithLogitsLoss with pos_weight = n_neg / n_pos (per ticker) |
| Metrics | Accuracy, macro-F1, AUC-ROC (train + val + test); aggregated as mean ± std across 5 tickers |
| Raw predictions saved | `results/predictions/<model>_<ticker>.npz` — probs, labels, dates, tickers |
| No-news days | Zero vector |
| Text dim | FinBERT 768-d → PCA → 64-d |

## Figures

**Fig 1 — EDA: feature distributions and class balance per ticker**

![EDA](figures/fig1_eda.png)

**Fig 2 — Model architecture: Self-Attention LSTM**

![Architecture](figures/fig2_architecture.png)

**Fig 3 — ROC curves and AUC bar chart (val vs test)**

![ROC and AUC](figures/fig3_roc_auc.png)

**Fig 4 — Error analysis: per-model summary and per-ticker AUC heatmap**

![Error analysis](figures/fig4_error_analysis.png)

**Fig 5 — Self-attention heatmap for best ticker (QCOM, 2019-02-21)**

![Attention heatmap](figures/fig5_attention_heatmap.png)

## Results

Aggregated test-set performance (mean ± std across 5 tickers, 3 seeds):

| Model | Test AUC | Test F1 | Test Acc |
|---|---|---|---|
| LR (B0) | 0.510 ± 0.017 | 0.505 ± 0.017 | 0.509 ± 0.014 |
| RF-notxt (B1) | 0.502 ± 0.014 | 0.490 ± 0.031 | 0.513 ± 0.010 |
| RF-full (B2) | 0.510 ± 0.013 | 0.485 ± 0.018 | 0.511 ± 0.022 |
| LSTM (A1) | 0.492 ± 0.016 | 0.393 ± 0.052 | 0.506 ± 0.007 |
| SelfAtt-notxt (A2) | 0.496 ± 0.036 | 0.351 ± 0.034 | 0.496 ± 0.016 |
| SelfAtt-full (M) | 0.503 ± 0.013 | 0.464 ± 0.028 | 0.501 ± 0.013 |

All models achieve test AUC within ±0.02 of the 0.50 random baseline, consistent with the Efficient Market Hypothesis. Key findings:

- **LR matches or outperforms all neural models** on test AUC — temporal modelling adds no generalisation benefit on this task.
- **RF severely overfits** (train AUC ≈ 1.0, test AUC ≈ 0.50) regardless of text inclusion.
- **Neural models partially collapse** toward majority-class prediction (F1 < 0.50), most severely in SelfAtt-notxt (F1 = 0.35). SelfAtt-full recovers F1 to 0.46, suggesting text features provide soft regularisation.
- **MRK is consistently the hardest ticker** across all models; QCOM and JNJ are relatively easier but still near-random.
- **Val AUC > Test AUC for every model** — Optuna-tuned patterns do not generalise from 2018 to the 2019–2020 test period.
