#!/usr/bin/env python3
"""Train a tabular VAE and export latent embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import project_root  # noqa: E402
from models.vae import VAEConfig, TabularVAE, vae_loss  # noqa: E402
from training.train_utils import (  # noqa: E402
    default_vae_paths,
    default_vae_prediction_paths,
    ensure_parent,
    latent_prediction_frame,
    load_model_input,
    set_global_seed,
    split_model_input,
    write_json,
)


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def _load_config(path: Path | None, *, input_dim: int) -> VAEConfig:
    base = VAEConfig(input_dim=input_dim)
    if path is None:
        return base
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    hidden_dims = tuple(payload.get("hidden_dims", list(base.hidden_dims)))
    return VAEConfig(
        input_dim=input_dim,
        latent_dim=int(payload.get("latent_dim", base.latent_dim)),
        hidden_dims=hidden_dims,
        learning_rate=float(payload.get("learning_rate", base.learning_rate)),
        batch_size=int(payload.get("batch_size", base.batch_size)),
        epochs=int(payload.get("epochs", base.epochs)),
        beta=float(payload.get("beta", base.beta)),
        seed=int(payload.get("seed", base.seed)),
        device=str(payload.get("device", base.device)),
    )


def _make_loader(features: pd.DataFrame, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    tensor = torch.tensor(features.to_numpy(dtype="float32"), dtype=torch.float32)
    dataset = TensorDataset(tensor)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _evaluate_epoch(model: TabularVAE, loader: DataLoader, *, beta: float, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
    batches = 0
    with torch.no_grad():
        for (batch,) in loader:
            inputs = batch.to(device)
            reconstruction, mu, logvar = model(inputs)
            loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction,
                inputs,
                mu,
                logvar,
                beta=beta,
            )
            total_loss += float(loss.item())
            total_reconstruction += float(reconstruction_loss.item())
            total_kl += float(kl_loss.item())
            batches += 1
    if batches == 0:
        return {"loss": 0.0, "reconstruction_loss": 0.0, "kl_loss": 0.0}
    return {
        "loss": total_loss / batches,
        "reconstruction_loss": total_reconstruction / batches,
        "kl_loss": total_kl / batches,
    }


def _encode_latent(model: TabularVAE, features: pd.DataFrame, *, device: torch.device) -> pd.DataFrame:
    tensor = torch.tensor(features.to_numpy(dtype="float32"), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        latent = model.latent_mean(tensor).cpu().numpy()
    return pd.DataFrame(latent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a VAE on tabular fraud features and export latent embeddings.",
    )
    parser.add_argument("--train-input", type=Path, default=None, help="Optional train_tabular.parquet path.")
    parser.add_argument("--valid-input", type=Path, default=None, help="Optional valid_tabular.parquet path.")
    parser.add_argument("--test-input", type=Path, default=None, help="Optional test_tabular.parquet path.")
    parser.add_argument("--config-input", type=Path, default=None, help="Optional JSON config override for VAE training.")
    parser.add_argument("--config-output", type=Path, default=None, help="Optional output path for saved config.json.")
    parser.add_argument("--weights-output", type=Path, default=None, help="Optional output path for weights.pt.")
    parser.add_argument("--metrics-output", type=Path, default=None, help="Optional output path for metrics.json.")
    parser.add_argument("--latent-train-output", type=Path, default=None, help="Optional output path for latent train parquet.")
    parser.add_argument("--latent-valid-output", type=Path, default=None, help="Optional output path for latent valid parquet.")
    parser.add_argument("--latent-test-output", type=Path, default=None, help="Optional output path for latent test parquet.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    default_train, default_valid, default_test, default_config, default_weights, default_metrics = default_vae_paths()
    default_latent_train, default_latent_valid, default_latent_test = default_vae_prediction_paths()

    train_frame = load_model_input(_resolve_path(args.train_input, default_train))
    valid_frame = load_model_input(_resolve_path(args.valid_input, default_valid))
    test_frame = load_model_input(_resolve_path(args.test_input, default_test))

    train_ids, train_labels, train_features = split_model_input(train_frame)
    valid_ids, valid_labels, valid_features = split_model_input(valid_frame)
    test_ids, test_labels, test_features = split_model_input(test_frame)

    config = _load_config(args.config_input, input_dim=train_features.shape[1])
    set_global_seed(config.seed)
    requested_device = config.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    config = replace(config, device=requested_device)
    device = torch.device(config.device)
    model = TabularVAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_loader = _make_loader(train_features, batch_size=config.batch_size, shuffle=True, seed=config.seed)
    valid_loader = _make_loader(valid_features, batch_size=config.batch_size, shuffle=False, seed=config.seed)
    test_loader = _make_loader(test_features, batch_size=config.batch_size, shuffle=False, seed=config.seed)

    train_history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_reconstruction = 0.0
        epoch_kl = 0.0
        batches = 0
        for (batch,) in train_loader:
            inputs = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, logvar = model(inputs)
            loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction,
                inputs,
                mu,
                logvar,
                beta=config.beta,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_reconstruction += float(reconstruction_loss.item())
            epoch_kl += float(kl_loss.item())
            batches += 1
        train_history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss / max(batches, 1),
                "reconstruction_loss": epoch_reconstruction / max(batches, 1),
                "kl_loss": epoch_kl / max(batches, 1),
            }
        )

    valid_metrics = _evaluate_epoch(model, valid_loader, beta=config.beta, device=device)
    test_metrics = _evaluate_epoch(model, test_loader, beta=config.beta, device=device)
    final_train_metrics = train_history[-1] if train_history else {"loss": 0.0, "reconstruction_loss": 0.0, "kl_loss": 0.0}

    latent_train = _encode_latent(model, train_features, device=device).to_numpy()
    latent_valid = _encode_latent(model, valid_features, device=device).to_numpy()
    latent_test = _encode_latent(model, test_features, device=device).to_numpy()

    latent_train_frame = latent_prediction_frame(transaction_ids=train_ids, labels=train_labels, latent=latent_train)
    latent_valid_frame = latent_prediction_frame(transaction_ids=valid_ids, labels=valid_labels, latent=latent_valid)
    latent_test_frame = latent_prediction_frame(transaction_ids=test_ids, labels=test_labels, latent=latent_test)

    config_output = _resolve_path(args.config_output, default_config)
    weights_output = _resolve_path(args.weights_output, default_weights)
    metrics_output = _resolve_path(args.metrics_output, default_metrics)
    latent_train_output = _resolve_path(args.latent_train_output, default_latent_train)
    latent_valid_output = _resolve_path(args.latent_valid_output, default_latent_valid)
    latent_test_output = _resolve_path(args.latent_test_output, default_latent_test)

    write_json(ensure_parent(config_output), config.to_payload())
    ensure_parent(weights_output)
    torch.save(model.state_dict(), weights_output)
    metrics_payload = {
        "config": config.to_payload(),
        "train_final": final_train_metrics,
        "valid": valid_metrics,
        "test": test_metrics,
        "latent_dim": config.latent_dim,
        "feature_count": train_features.shape[1],
    }
    write_json(ensure_parent(metrics_output), metrics_payload)

    ensure_parent(latent_train_output)
    ensure_parent(latent_valid_output)
    ensure_parent(latent_test_output)
    latent_train_frame.to_parquet(latent_train_output, index=False)
    latent_valid_frame.to_parquet(latent_valid_output, index=False)
    latent_test_frame.to_parquet(latent_test_output, index=False)

    print(
        f"latent_dim={config.latent_dim} "
        f"epochs={config.epochs} "
        f"valid_loss={valid_metrics['loss']:.6f} "
        f"test_loss={test_metrics['loss']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
