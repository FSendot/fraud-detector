"""
API REST mínima para el dashboard (demo).
Solo biblioteca estándar de Python 3 — no requiere pip ni requirements.txt.

Ejecutar en EC2: python3 app.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from urllib.parse import parse_qs

from wsgiref.simple_server import make_server

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

_CORS_HEADERS = [
    ("Access-Control-Allow-Origin", "*"),
    ("Access-Control-Allow-Methods", "GET, OPTIONS"),
    ("Access-Control-Allow-Headers", "Content-Type, Accept"),
]


def _json_response(
    start_response, status: str, obj: object
) -> list[bytes]:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ] + _CORS_HEADERS
    start_response(status, headers)
    return [body]


def application(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET")
    path = environ.get("PATH_INFO") or "/"
    query = environ.get("QUERY_STRING") or ""

    if method == "OPTIONS":
        start_response("204 No Content", _CORS_HEADERS)
        return [b""]

    if method != "GET":
        return _json_response(
            start_response,
            "405 Method Not Allowed",
            {"error": "method_not_allowed"},
        )

    # "/" responde OK: muchos health checks (ALB/ELB) usan GET / por defecto.
    if path in ("/", "/health"):
        return _json_response(start_response, "200 OK", {"status": "ok"})

    if path == "/api/stats":
        return _json_response(start_response, "200 OK", _MOCK_STATS)

    if path == "/api/transactions":
        qs = parse_qs(query)
        raw_limit = (qs.get("limit") or ["20"])[0]
        try:
            limit = int(raw_limit)
        except ValueError:
            limit = 20
        limit = max(1, min(limit, 100))
        return _json_response(
            start_response,
            "200 OK",
            {"items": _MOCK_TRANSACTIONS[:limit]},
        )

    if path == "/api/version":
        return _json_response(
            start_response,
            "200 OK",
            {
                "service": "fraud-dashboard-api",
                "version": "1.0.0",
                "time": datetime.now(timezone.utc).isoformat(),
            },
        )

    return _json_response(
        start_response,
        "404 Not Found",
        {"error": "not_found", "path": path},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "0.0.0.0")
    with make_server(host, port, application) as httpd:
        print(f"Serving on http://{host}:{port}")
        httpd.serve_forever()
