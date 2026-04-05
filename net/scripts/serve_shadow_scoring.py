#!/usr/bin/env python3
"""Serve or execute shadow-only scoring against the packaged fraud bundle."""

from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import load_paths_config, project_root  # noqa: E402
from serving.scoring_service import ShadowScoringService, load_request_payload  # noqa: E402
from training.train_utils import write_json  # noqa: E402


def _resolve_path(candidate: Path | None, fallback: Path) -> Path:
    if candidate is None:
        return fallback
    expanded = candidate.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root() / expanded).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a shadow-only scoring adapter around the packaged fraud model bundle.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Bundle directory or manifest path. Defaults to artifacts/bundles/model_v1.",
    )
    parser.add_argument(
        "--request-file",
        type=Path,
        default=None,
        help="Optional JSON request file for one-shot scoring. If omitted, the script serves HTTP.",
    )
    parser.add_argument(
        "--response-output",
        type=Path,
        default=None,
        help="Optional JSON output path for one-shot scoring.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="HTTP host for service mode.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port for service mode.")
    return parser


def _write_json_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict[str, object]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    handler.send_response(int(status))
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _handler_factory(service: ShadowScoringService):
    class ShadowHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                _write_json_response(
                    self,
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "shadow_mode": True,
                        "model_version": service.bundle_version,
                    },
                )
                return
            _write_json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/score-shadow":
                _write_json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length)
                payload = json.loads(raw_body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                response = service.score_payload(payload)
            except Exception as exc:  # noqa: BLE001
                _write_json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {
                        "shadow_mode": True,
                        "error": str(exc),
                    },
                )
                return
            _write_json_response(self, HTTPStatus.OK, response)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    return ShadowHandler


def main() -> int:
    args = build_parser().parse_args()
    paths = load_paths_config()
    bundle_path = _resolve_path(args.bundle, paths["artifact_bundles_dir"] / "model_v1")
    service = ShadowScoringService(bundle_path=bundle_path)

    if args.request_file is not None:
        payload = load_request_payload(_resolve_path(args.request_file, args.request_file))
        response = service.score_payload(payload)
        if args.response_output is not None:
            write_json(_resolve_path(args.response_output, args.response_output), response)
            print(f"response={_resolve_path(args.response_output, args.response_output)}")
        else:
            print(json.dumps(response, indent=2, sort_keys=True))
        return 0

    server = ThreadingHTTPServer((args.host, args.port), _handler_factory(service))
    print(f"shadow_scoring_service=http://{args.host}:{args.port} model_version={service.bundle_version}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
