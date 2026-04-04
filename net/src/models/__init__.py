"""Model definitions for fraud detection experiments."""

from .nystrom_gp import NystromGPConfig, fit_nystrom_classifier
from .vae import VAEConfig, TabularVAE

__all__ = [
    "NystromGPConfig",
    "VAEConfig",
    "TabularVAE",
    "fit_nystrom_classifier",
]

