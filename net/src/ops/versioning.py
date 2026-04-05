"""Bundle version selection and rollback helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.config import DEFAULT_CONFIG_DIR, load_yaml_file


DEFAULT_RETRAINING_PATH = DEFAULT_CONFIG_DIR / "retraining.yaml"
VERSION_PATTERN = re.compile(r"model_v(\d+)$")


@dataclass(frozen=True)
class BundleVersionRecord:
    """Registered bundle version and its readiness metadata."""

    version: str
    manifest_path: Path
    validation_hashes_verified: bool
    operational_decision_variant: str
    calibrated_threshold: float | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "manifest_path": str(self.manifest_path),
            "validation_hashes_verified": self.validation_hashes_verified,
            "operational_decision_variant": self.operational_decision_variant,
            "calibrated_threshold": self.calibrated_threshold,
        }


@dataclass(frozen=True)
class VersionDecision:
    """Selection or rollback recommendation."""

    active_version: str | None
    selected_version: str
    reason: str
    rollback_triggered: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "active_version": self.active_version,
            "selected_version": self.selected_version,
            "reason": self.reason,
            "rollback_triggered": self.rollback_triggered,
        }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def load_retraining_config(path: Path | None = None) -> dict[str, Any]:
    """Load retraining/rollback thresholds."""

    config_path = DEFAULT_RETRAINING_PATH if path is None else path
    payload = load_yaml_file(config_path)
    retraining = payload.get("retraining", {})
    if not isinstance(retraining, dict):
        raise ValueError("expected 'retraining' mapping in retraining config")
    return retraining


def discover_bundle_versions(bundle_root: Path) -> list[BundleVersionRecord]:
    """Discover bundle manifests and derive selection metadata."""

    records: list[BundleVersionRecord] = []
    for manifest_path in sorted(bundle_root.glob("model_v*/manifest.json")):
        payload = _load_json(manifest_path)
        version = str(payload.get("bundle_version", manifest_path.parent.name))
        runtime = payload.get("runtime_metadata", {})
        operational = runtime.get("operational_defaults", {})
        records.append(
            BundleVersionRecord(
                version=version,
                manifest_path=manifest_path,
                validation_hashes_verified=bool(payload.get("validation", {}).get("hashes_verified", False)),
                operational_decision_variant=str(operational.get("decision_variant", "unknown")),
                calibrated_threshold=(
                    float(operational["decision_threshold"])
                    if isinstance(operational.get("decision_threshold"), (int, float))
                    else None
                ),
            )
        )
    records.sort(key=lambda record: _version_sort_key(record.version), reverse=True)
    return records


def _version_sort_key(version: str) -> tuple[int, str]:
    match = VERSION_PATTERN.match(version)
    if match:
        return (int(match.group(1)), version)
    return (-1, version)


def select_bundle_version(
    bundle_root: Path,
    *,
    requested_version: str | None = None,
    active_version: str | None = None,
) -> VersionDecision:
    """Choose the version to serve when no rollback is required."""

    records = discover_bundle_versions(bundle_root)
    if not records:
        raise FileNotFoundError(f"no bundle manifests found in {bundle_root}")

    indexed = {record.version: record for record in records}
    if requested_version is not None:
        if requested_version not in indexed:
            raise ValueError(f"requested bundle version not found: {requested_version}")
        selected = indexed[requested_version]
        return VersionDecision(
            active_version=active_version,
            selected_version=selected.version,
            reason="explicit_request",
            rollback_triggered=False,
        )

    for record in records:
        if record.validation_hashes_verified:
            return VersionDecision(
                active_version=active_version,
                selected_version=record.version,
                reason="latest_validated_bundle",
                rollback_triggered=False,
            )
    raise ValueError("no hash-verified bundle versions are available")


def should_trigger_rollback(
    *,
    active_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    config: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Compare active-vs-baseline metrics against rollback thresholds."""

    rollback_cfg = config.get("rollback", {})
    if not bool(rollback_cfg.get("enabled", True)):
        return False, []
    thresholds = rollback_cfg.get("critical_metric_drop_absolute", {})
    reasons: list[str] = []
    for metric_name, threshold in thresholds.items():
        if metric_name not in active_metrics or metric_name not in baseline_metrics:
            continue
        drop = float(baseline_metrics[metric_name]) - float(active_metrics[metric_name])
        if drop >= float(threshold):
            reasons.append(f"{metric_name}_drop={drop:.6f}")
    return bool(reasons), reasons


def choose_rollback_version(
    bundle_root: Path,
    *,
    active_version: str,
    active_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    retraining_config: dict[str, Any],
) -> VersionDecision:
    """Choose whether to stay on the active bundle or roll back to the prior one."""

    rollback, reasons = should_trigger_rollback(
        active_metrics=active_metrics,
        baseline_metrics=baseline_metrics,
        config=retraining_config,
    )
    if not rollback:
        return VersionDecision(
            active_version=active_version,
            selected_version=active_version,
            reason="rollback_not_triggered",
            rollback_triggered=False,
        )

    records = discover_bundle_versions(bundle_root)
    active_index = next((index for index, record in enumerate(records) if record.version == active_version), None)
    if active_index is None:
        raise ValueError(f"active version not found in bundle registry: {active_version}")
    for candidate in records[active_index + 1 :]:
        if candidate.validation_hashes_verified:
            return VersionDecision(
                active_version=active_version,
                selected_version=candidate.version,
                reason=";".join(reasons),
                rollback_triggered=True,
            )
    return VersionDecision(
        active_version=active_version,
        selected_version=active_version,
        reason="rollback_triggered_but_no_prior_valid_bundle",
        rollback_triggered=False,
    )
