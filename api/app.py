"""
API REST mínima para el dashboard (demo).
Ejecutar en EC2: python app.py  (escucha en 0.0.0.0:5000)
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Datos de ejemplo; reemplazá por consultas a Redshift/DB cuando tengas el pipeline.
_MOCK_STATS = {
    "total_transactions": 12847,
    "allowed": 11200,
    "challenge": 1021,
    "blocked": 626,
}

_MOCK_TRANSACTIONS = [
    {
        "transaction_id": "tx_demo_001",
        "user_id": "u_100",
        "amount": 15000.0,
        "currency": "ARS",
        "country": "BR",
        "score": 85,
        "decision": "blocked",
        "processed_at": "2026-04-03T10:22:00.123Z",
    },
    {
        "transaction_id": "tx_demo_002",
        "user_id": "u_101",
        "amount": 3200.0,
        "currency": "ARS",
        "country": "AR",
        "score": 35,
        "decision": "allowed",
        "processed_at": "2026-04-03T10:25:11.000Z",
    },
    {
        "transaction_id": "tx_demo_003",
        "user_id": "u_102",
        "amount": 8900.0,
        "currency": "ARS",
        "country": "UY",
        "score": 55,
        "decision": "challenge",
        "processed_at": "2026-04-03T10:28:44.500Z",
    },
]


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/stats")
def stats():
    return jsonify(_MOCK_STATS)


@app.get("/api/transactions")
def transactions():
    limit = request.args.get("limit", default="20", type=int)
    limit = max(1, min(limit, 100))
    return jsonify({"items": _MOCK_TRANSACTIONS[:limit]})


@app.get("/api/version")
def version():
    return jsonify(
        {
            "service": "fraud-dashboard-api",
            "version": "1.0.0",
            "time": datetime.now(timezone.utc).isoformat(),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
