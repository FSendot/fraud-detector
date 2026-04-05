# Serving Contracts

This directory will hold the schema, feature-order, and model-metadata
contracts shared between the Python training pipeline and the future Go
serving layer.

Current files:

- `scoring_request.json` defines the shadow scoring request payload.
- `scoring_response.json` defines the shadow scoring response payload.

These contracts are intentionally shadow-only for now. They are suitable for a
future Go client that wants to call the Python scorer for offline comparison
without triggering any live fraud decisions.
