"""PyTorch VAE for tabular fraud features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class VAEConfig:
    """Configuration for the tabular VAE."""

    input_dim: int
    latent_dim: int = 8
    hidden_dims: tuple[int, ...] = (64, 32)
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 20
    beta: float = 0.1
    seed: int = 7
    device: str = "cpu"
    logvar_min: float = -10.0
    logvar_max: float = 10.0

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


class TabularVAE(nn.Module):
    """Small multilayer perceptron VAE for tabular inputs."""

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder_backbone = self._build_mlp(
            input_dim=config.input_dim,
            layer_dims=config.hidden_dims,
        )
        encoder_output_dim = config.hidden_dims[-1] if config.hidden_dims else config.input_dim
        self.encoder_mu = nn.Linear(encoder_output_dim, config.latent_dim)
        self.encoder_logvar = nn.Linear(encoder_output_dim, config.latent_dim)

        decoder_dims = tuple(reversed(config.hidden_dims))
        self.decoder_backbone = self._build_mlp(
            input_dim=config.latent_dim,
            layer_dims=decoder_dims,
        )
        decoder_output_dim = decoder_dims[-1] if decoder_dims else config.latent_dim
        self.decoder_out = nn.Linear(decoder_output_dim, config.input_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _build_mlp(*, input_dim: int, layer_dims: tuple[int, ...]) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in layer_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        if not layers:
            layers.append(nn.Identity())
        return nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder_backbone(inputs)
        mu = self.encoder_mu(hidden)
        logvar = self.encoder_logvar(hidden).clamp(min=self.config.logvar_min, max=self.config.logvar_max)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_backbone(latent)
        return self.decoder_out(hidden)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(inputs)
        latent = self.reparameterize(mu, logvar)
        reconstruction = self.decode(latent)
        return reconstruction, mu, logvar

    def latent_mean(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return deterministic latent means for downstream reuse."""

        mu, _ = self.encode(inputs)
        return mu


def vae_loss(
    reconstruction: torch.Tensor,
    inputs: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, reconstruction, and KL losses."""

    reconstruction_loss = nn.functional.mse_loss(reconstruction, inputs, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss
