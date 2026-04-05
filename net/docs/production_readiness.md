# Production Readiness Layer

This document defines the operational contract around the packaged fraud model.
It does not deploy cloud infrastructure, but it makes the model bundle ready to
be integrated into one.

## Default Serving Posture

- Serve the packaged `fused_calibrated` score from the active bundle.
- Keep the scorer in shadow mode until policy and monitoring are proven stable.
- Treat policy outputs as recommendations until the rollout owner explicitly
  enables side effects in the outer service.

## Threshold Policy

The default policy file is
[production_thresholds.yaml](fraud-detector/net/configs/production_thresholds.yaml).

Default actions:

- `allow`: score `< 0.15`
- `review`: `0.15 <= score < 0.55`
- `block`: score `>= 0.55`

Recommended rollout path:

1. Start with `allow` and `review` only in production logs.
2. Compare shadow recommendations against existing risk decisions.
3. Promote `review` routing first.
4. Promote `block` only after business review confirms false-positive tolerance.

## Monitoring

The default monitoring file is
[monitoring.yaml](fraud-detector/net/configs/monitoring.yaml).

The readiness layer expects the future platform to track:

- score drift on `raw_fused_score` and `calibrated_score`
- feature drift on selected high-signal fields
- latency `p50`, `p95`, and `p99`
- contract mismatches, rebuild rate, duplicate ids, and missing features

Severity guidance:

- warning: investigate during business hours
- critical: freeze promotions, review active version health, and evaluate rollback

## Retraining

The default retraining file is
[retraining.yaml](fraud-detector/net/configs/retraining.yaml).

Retraining should be considered when one or more of these happen:

- score drift exceeds the configured PSI trigger
- feature drift exceeds the configured PSI trigger
- precision or recall drops materially against the active baseline
- data quality breaches indicate the model no longer matches production input
- new labeled data volume is large enough to justify a refresh

Promotion gate:

- candidate bundle must outperform the active bundle on the configured offline
  metrics
- candidate bundle must pass shadow evaluation before it becomes active

## Version Selection And Rollback

Version selection logic lives in
[versioning.py](fraud-detector/net/src/ops/versioning.py).

Expected behavior:

- choose the latest hash-verified bundle by default
- allow explicit version pinning for canaries or investigations
- roll back only to a previously verified bundle
- trigger rollback when active metrics fall below baseline by the configured
  critical absolute margins

The bundle manifest is the source of truth for:

- bundle version
- packaged thresholds
- fusion runtime selection
- calibration method
- artifact integrity hashes

## Integration Expectations

The future service layer should:

- load the active bundle manifest once at startup
- expose the active bundle version in health and scoring responses
- log policy recommendation, not only final label
- persist enough shadow telemetry to recompute drift and threshold reports
- support fast rollback by switching the active bundle pointer, not by rebuilding
  artifacts in place

## Suggested Rollout Checklist

1. Validate the input dataset against the feature contract.
2. Run batch inference on a recent labeled backfill.
3. Confirm the monitoring baselines for score and feature drift.
4. Deploy the shadow scorer behind the integration boundary.
5. Compare shadow outputs to the incumbent system for at least one stable window.
6. Promote `review` routing first, then consider `block`.
