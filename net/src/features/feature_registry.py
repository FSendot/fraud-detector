"""Feature metadata registry for machine-readable documentation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeatureSpec:
    """Machine-readable description for one feature column."""

    name: str
    group: str
    dtype: str
    source: str
    point_in_time_safe: bool
    description: str


@dataclass(frozen=True)
class FeatureRegistry:
    """Container for source and derived feature metadata."""

    dataset_name: str
    source_features: tuple[FeatureSpec, ...]
    derived_features: tuple[FeatureSpec, ...]

    def to_payload(self) -> dict[str, Any]:
        """Return a stable JSON-serializable payload."""

        all_features = [*self.source_features, *self.derived_features]
        return {
            "dataset_name": self.dataset_name,
            "feature_count": len(all_features),
            "source_features": [asdict(feature) for feature in self.source_features],
            "derived_features": [asdict(feature) for feature in self.derived_features],
        }


def write_feature_registry(path: Path, registry: FeatureRegistry) -> None:
    """Persist the feature registry to JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(registry.to_payload(), handle, indent=2, sort_keys=True)
        handle.write("\n")

