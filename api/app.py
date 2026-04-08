from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from decimal import Decimal
from urllib.parse import parse_qs

from wsgiref.simple_server import make_server

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    _HAVE_PSYCOPG2 = True
except ImportError:
    _HAVE_PSYCOPG2 = False

# ---------- RDS: editá aquí (sin .env). Si DB_HOST queda vacío, se usa el mock. ----------
DB_HOST = ""
DB_PORT = 5432
DB_NAME = ""
DB_USER = ""
DB_PASSWORD = ""
DB_SSLMODE = "require"

# Datos de ejemplo si RDS no está configurado
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

_SQL_STATS = """
SELECT
  COUNT(*)::bigint AS total_transactions,
  COUNT(*) FILTER (WHERE LOWER(TRIM(decision)) = 'allowed')::bigint AS allowed,
  COUNT(*) FILTER (WHERE LOWER(TRIM(decision)) = 'challenge')::bigint AS challenge,
  COUNT(*) FILTER (WHERE LOWER(TRIM(decision)) = 'blocked')::bigint AS blocked
FROM transactions;
"""

_SQL_TRANSACTIONS = """
SELECT
  transaction_id,
  user_id,
  person_id,
  account_id,
  amount,
  currency,
  "timestamp",
  channel,
  destination_account,
  country,
  score,
  decision,
  flag_amount,
  flag_country,
  flag_velocity,
  flag_destination,
  profile_avg_amount,
  profile_std_dev,
  profile_typical_countries,
  profile_tx_last_10min,
  model_version,
  calibrated_score,
  processor_version,
  processed_at
FROM transactions
ORDER BY processed_at DESC NULLS LAST, transaction_id DESC
LIMIT %s;
"""


def _db_env_configured() -> bool:
    return bool(DB_HOST and DB_NAME and DB_USER)


def _connect_rds():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode=DB_SSLMODE,
    )


def _json_value(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def _row_to_api_dict(row: dict) -> dict:
    return {k: _json_value(v) for k, v in row.items()}


def _fetch_stats_from_db():
    with _connect_rds() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(_SQL_STATS)
            row = cur.fetchone()
    if not row:
        return {
            "total_transactions": 0,
            "allowed": 0,
            "challenge": 0,
            "blocked": 0,
        }
    return {
        "total_transactions": int(row["total_transactions"]),
        "allowed": int(row["allowed"]),
        "challenge": int(row["challenge"]),
        "blocked": int(row["blocked"]),
    }


def _fetch_transactions_from_db(limit: int):
    with _connect_rds() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(_SQL_TRANSACTIONS, (limit,))
            rows = cur.fetchall()
    return [_row_to_api_dict(dict(r)) for r in rows]


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
        if _db_env_configured():
            if not _HAVE_PSYCOPG2:
                return _json_response(
                    start_response,
                    "503 Service Unavailable",
                    {
                        "error": "psycopg2_missing",
                        "detail": "Instalá python3-psycopg2 con el gestor de paquetes del SO.",
                    },
                )
            try:
                stats = _fetch_stats_from_db()
            except Exception as e:
                return _json_response(
                    start_response,
                    "503 Service Unavailable",
                    {"error": "database_error", "detail": str(e)},
                )
            return _json_response(start_response, "200 OK", stats)
        return _json_response(start_response, "200 OK", _MOCK_STATS)

    if path == "/api/transactions":
        qs = parse_qs(query)
        raw_limit = (qs.get("limit") or ["20"])[0]
        try:
            limit = int(raw_limit)
        except ValueError:
            limit = 20
        limit = max(1, min(limit, 100))
        if _db_env_configured():
            if not _HAVE_PSYCOPG2:
                return _json_response(
                    start_response,
                    "503 Service Unavailable",
                    {
                        "error": "psycopg2_missing",
                        "detail": "Instalá python3-psycopg2 con el gestor de paquetes del SO.",
                    },
                )
            try:
                items = _fetch_transactions_from_db(limit)
            except Exception as e:
                return _json_response(
                    start_response,
                    "503 Service Unavailable",
                    {"error": "database_error", "detail": str(e)},
                )
            return _json_response(start_response, "200 OK", {"items": items})
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
