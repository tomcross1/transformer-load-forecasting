from __future__ import annotations

import math

import torch
from torch import nn


class RecurrentForecaster(nn.Module):
    def __init__(self, n_features: int, horizon: int, cell: str = "lstm",
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(n_features, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.head(output[:, -1])


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerForecaster(nn.Module):
    """Encoder-only Transformer with direct multi-horizon regression head."""

    def __init__(self, n_features: int, horizon: int, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 3, dim_feedforward: int = 128,
                 dropout: float = 0.1, positional_encoding: bool = True):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.position = SinusoidalPositionalEncoding(d_model) if positional_encoding else nn.Identity()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.position(self.input_projection(x))
        z = self.encoder(z)
        return self.head(z[:, -1])


def build_model(name: str, n_features: int, horizon: int, cfg: dict,
                positional_encoding: bool = True) -> nn.Module:
    if name in {"lstm", "gru"}:
        return RecurrentForecaster(n_features, horizon, cell=name,
                                   hidden_size=cfg.get("d_model", 64),
                                   num_layers=cfg.get("num_layers", 2),
                                   dropout=cfg.get("dropout", 0.1))
    if name == "transformer":
        return TransformerForecaster(n_features, horizon, positional_encoding=positional_encoding, **cfg)
    raise ValueError(f"Unknown neural model: {name}")

