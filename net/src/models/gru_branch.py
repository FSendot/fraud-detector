"""GRU branch model for sequence-based fraud classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from torch import nn


@dataclass(frozen=True)
class GRUBranchConfig:
    """Configuration for the sequence GRU fraud branch."""

    sequence_input_dim: int
    hidden_dim: int = 64
    gru_layers: int = 1
    classifier_hidden_dim: int = 32
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 10
    patience: int = 3
    downsample_ratio: float = 3.0
    max_train_samples: int | None = None
    max_valid_monitor_samples: int = 200000
    seed: int = 7
    device: str = "cpu"
    pos_weight: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


class GRUFraudClassifier(nn.Module):
    """Sequence-only GRU classifier with a small MLP head."""

    def __init__(self, config: GRUBranchConfig) -> None:
        super().__init__()
        gru_dropout = config.dropout if config.gru_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=config.sequence_input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, 1),
        )

    def forward(self, x_seq):
        _, hidden = self.gru(x_seq)
        last_hidden = hidden[-1]
        return self.classifier(last_hidden).squeeze(-1)
