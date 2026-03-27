"""
models.py
---------
Model architecture definitions for FinBERT-Attention-LSTM.

Models
------
1. LogisticRegressionModel  — linear baseline (required by coursework)
2. LSTMModel                — plain LSTM, price features only
3. SelfAttLSTM              — multi-head self-attention + LSTM
                              used for both ablation (price only) and
                              main model (price + FinBERT PCA-64)

Architecture: Self-Att-LSTM
---------------------------
Input sequence X ∈ R^(T × d_in)
  → Multi-Head Self-Attention → context-weighted X' ∈ R^(T × d_in)
  → Dropout
  → LSTM (hidden=64)          → last hidden state h_T ∈ R^64
  → Dropout
  → Linear(64 → 1) + Sigmoid  → P(next-day return > 0)

where d_in = n_price_features (ablation) or n_price + n_text_pca (full model).
"""

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression  # kept for API consistency


# ---------------------------------------------------------------------------
# Constants (kept in sync with data_pipeline.py / dataset.py)
# ---------------------------------------------------------------------------
N_PRICE_FEATURES = 13  # 9 technical indicators + 4 OHLCV candlestick features
N_TEXT_FEATURES  = 64  # PCA-reduced FinBERT dim
N_FULL_FEATURES  = N_PRICE_FEATURES + N_TEXT_FEATURES  # 77


# ---------------------------------------------------------------------------
# 1. Logistic Regression (sklearn wrapper)
# ---------------------------------------------------------------------------
class LogisticRegressionModel:
    """
    Thin wrapper around sklearn LogisticRegression.
    Fits on flattened sliding-window price features.
    Input X: (N, T * n_price_features)  — flattened window
    """
    def __init__(self, class_weight="balanced", max_iter=1000, random_state=42):
        self.model = LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            self.model = pickle.load(f)


# ---------------------------------------------------------------------------
# 2. Plain LSTM
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    """
    Single-layer LSTM on price features.
    Input:  (batch, T, n_price_features)
    Output: (batch, 1) logit — apply sigmoid for probability
    """
    def __init__(
        self,
        input_dim: int = N_PRICE_FEATURES,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, input_dim)
        _, (h_n, _) = self.lstm(x)      # h_n: (num_layers, batch, hidden)
        h_last = h_n[-1]                 # last layer: (batch, hidden)
        out = self.fc(self.dropout(h_last))  # (batch, 1)
        return out


# ---------------------------------------------------------------------------
# 3. Self-Attention + LSTM
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention over the time dimension.
    Input/output shape: (batch, T, d_model)

    Attention weights are stored in self.attn_weights after each forward pass
    for post-hoc visualisation (averaged across heads).
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_weights: torch.Tensor | None = None  # populated in forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention: Q = K = V = x
        out, weights = self.attn(x, x, x)
        self.attn_weights = weights.detach()  # (batch, T, T)
        return out


class SelfAttLSTM(nn.Module):
    """
    Self-Attention LSTM.

    Parameters
    ----------
    input_dim   : feature dimension per time step
                  = N_PRICE_FEATURES for ablation (price only)
                  = N_FULL_FEATURES  for main model (price + text)
    hidden_dim  : LSTM hidden units (default 64)
    n_heads     : attention heads (default 4)
    dropout     : dropout rate after attention and after LSTM (default 0.2)

    Forward
    -------
    x : (batch, T, input_dim)
    returns logit : (batch, 1)   — apply sigmoid for probability
    """
    def __init__(
        self,
        input_dim: int = N_PRICE_FEATURES,
        hidden_dim: int = 64,
        n_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Project input to a dimension divisible by n_heads if needed
        self.proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        d_model = hidden_dim

        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout=0.1)
        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, input_dim)
        x = self.proj(x)                         # (batch, T, d_model)
        attn_out = self.attention(x)              # (batch, T, d_model)
        x = self.layer_norm(x + self.attn_dropout(attn_out))  # residual

        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]                          # (batch, hidden_dim)
        out = self.fc(self.lstm_dropout(h_last))  # (batch, 1)
        return out

    @property
    def attn_weights(self) -> torch.Tensor | None:
        """Attention weights from last forward pass. Shape: (batch, T, T)."""
        return self.attention.attn_weights


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_model(name: str) -> nn.Module | LogisticRegressionModel:
    """
    name options:
        'logreg'              — LogisticRegressionModel
        'lstm'                — LSTMModel (price only)
        'selfattlstm_notxt'   — SelfAttLSTM (price only, ablation)
        'selfattlstm_full'    — SelfAttLSTM (price + text)
    """
    if name == "logreg":
        return LogisticRegressionModel()
    elif name == "lstm":
        return LSTMModel(input_dim=N_PRICE_FEATURES)
    elif name == "selfattlstm_notxt":
        return SelfAttLSTM(input_dim=N_PRICE_FEATURES, num_layers=1)
    elif name == "selfattlstm_full":
        return SelfAttLSTM(input_dim=N_FULL_FEATURES, num_layers=1)
    else:
        raise ValueError(f"Unknown model name: {name}")
