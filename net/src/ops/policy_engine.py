"""Production policy routing helpers for fraud scores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.config import DEFAULT_CONFIG_DIR, load_yaml_file


DEFAULT_POLICY_PATH = DEFAULT_CONFIG_DIR / "production_thresholds.yaml"


@dataclass(frozen=True)
class ActionRange:
    """Inclusive/exclusive score bounds for one production action."""

    name: str
    min_score_inclusive: float | None
    max_score_exclusive: float | None
    rationale: str

    def matches(self, score: float) -> bool:
        if self.min_score_inclusive is not None and score < self.min_score_inclusive:
            return False
        if self.max_score_exclusive is not None and score >= self.max_score_exclusive:
            return False
        return True


@dataclass(frozen=True)
class PolicyDecision:
    """Resolved policy decision for a scored transaction."""

    action: str
    score: float | None
    score_field: str
    reason: str
    shadow_only: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "score": self.score,
            "score_field": self.score_field,
            "reason": self.reason,
            "shadow_only": self.shadow_only,
        }


@dataclass(frozen=True)
class PolicyConfig:
    """Loaded production threshold policy."""

    score_field: str
    action_order: tuple[str, ...]
    actions: tuple[ActionRange, ...]
    on_missing_score: str
    on_contract_mismatch: str
    shadow_enabled: bool
    emit_decision_recommendation_only: bool
    log_branch_outputs: bool


def load_policy_config(path: Path | None = None) -> PolicyConfig:
    """Load the production threshold policy from YAML."""

    config_path = DEFAULT_POLICY_PATH if path is None else path
    payload = load_yaml_file(config_path)
    policy = payload.get("policy", {})
    if not isinstance(policy, dict):
        raise ValueError("expected 'policy' mapping in production_thresholds config")

    action_order = tuple(str(name) for name in policy.get("action_order", []))
    actions_payload = policy.get("actions", {})
    if not isinstance(actions_payload, dict):
        raise ValueError("expected 'actions' mapping in production_thresholds config")

    actions: list[ActionRange] = []
    for name in action_order:
        item = actions_payload.get(name, {})
        if not isinstance(item, dict):
            raise ValueError(f"expected action mapping for {name}")
        actions.append(
            ActionRange(
                name=name,
                min_score_inclusive=float(item["min_score_inclusive"]) if "min_score_inclusive" in item else None,
                max_score_exclusive=float(item["max_score_exclusive"]) if "max_score_exclusive" in item else None,
                rationale=str(item.get("rationale", "")),
            )
        )

    _validate_policy_ranges(actions)
    fallback = policy.get("fallback", {})
    shadow_defaults = policy.get("shadow_defaults", {})
    return PolicyConfig(
        score_field=str(policy.get("score_field", "calibrated_score")),
        action_order=action_order,
        actions=tuple(actions),
        on_missing_score=str(fallback.get("on_missing_score", "review")),
        on_contract_mismatch=str(fallback.get("on_contract_mismatch", "review")),
        shadow_enabled=bool(shadow_defaults.get("enabled", True)),
        emit_decision_recommendation_only=bool(shadow_defaults.get("emit_decision_recommendation_only", True)),
        log_branch_outputs=bool(shadow_defaults.get("log_branch_outputs", True)),
    )


def _validate_policy_ranges(actions: list[ActionRange]) -> None:
    previous_max: float | None = None
    for action in actions:
        if (
            action.min_score_inclusive is not None
            and action.max_score_exclusive is not None
            and action.min_score_inclusive >= action.max_score_exclusive
        ):
            raise ValueError(f"invalid range for action {action.name}")
        if previous_max is not None and action.min_score_inclusive is not None and action.min_score_inclusive < previous_max:
            raise ValueError("policy action ranges overlap or are out of order")
        if action.max_score_exclusive is not None:
            previous_max = action.max_score_exclusive


def decide_action(
    score_payload: dict[str, Any],
    *,
    contract_valid: bool,
    config: PolicyConfig,
) -> PolicyDecision:
    """Choose allow/review/block using the configured score field and fallbacks."""

    score_value = score_payload.get(config.score_field)
    if not contract_valid:
        return PolicyDecision(
            action=config.on_contract_mismatch,
            score=None if score_value is None else float(score_value),
            score_field=config.score_field,
            reason="contract_mismatch_fallback",
            shadow_only=config.shadow_enabled,
        )
    if score_value is None:
        return PolicyDecision(
            action=config.on_missing_score,
            score=None,
            score_field=config.score_field,
            reason="missing_score_fallback",
            shadow_only=config.shadow_enabled,
        )
    score = float(score_value)
    for action in config.actions:
        if action.matches(score):
            return PolicyDecision(
                action=action.name,
                score=score,
                score_field=config.score_field,
                reason=action.rationale,
                shadow_only=config.shadow_enabled,
            )
    raise ValueError(f"score {score} did not match any configured action range")
