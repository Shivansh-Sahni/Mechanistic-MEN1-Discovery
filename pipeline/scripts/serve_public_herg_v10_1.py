#!/usr/bin/env python3
"""Password-gated reverse proxy for sharing the local hERG V10.1 demo."""

from __future__ import annotations

import argparse
import base64
import hmac
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ALLOWED_PATHS = {"/", "/index.html", "/api/model-info", "/api/health", "/api/predict"}


def _authorization_value(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {token}"


def _handler(upstream: str, username: str, password: str) -> type[BaseHTTPRequestHandler]:
    expected = _authorization_value(username, password)

    class AuthenticatedProxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "hERGDemoProxy/1.0"
        allowed_paths: ClassVar[set[str]] = ALLOWED_PATHS

        def _authorized(self) -> bool:
            supplied = self.headers.get("authorization", "")
            return hmac.compare_digest(supplied, expected)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.send_header("x-content-type-options", "nosniff")
            self.send_header("x-frame-options", "DENY")
            self.send_header("referrer-policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _challenge(self) -> None:
            body = b"Authentication required"
            self.send_response(HTTPStatus.UNAUTHORIZED.value)
            self.send_header("www-authenticate", 'Basic realm="hERG research demo"')
            self.send_header("content-type", "text/plain; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _proxy(self) -> None:
            if not self._authorized():
                self._challenge()
                return
            path = self.path.split("?", 1)[0]
            if path not in self.allowed_paths:
                self._send(HTTPStatus.NOT_FOUND.value, b"Not found", "text/plain; charset=utf-8")
                return
            content_length = int(self.headers.get("content-length", "0"))
            if content_length < 0 or content_length > 100_000:
                self._send(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE.value,
                    b"Invalid request size",
                    "text/plain; charset=utf-8",
                )
                return
            body = self.rfile.read(content_length) if content_length else None
            request = Request(
                f"{upstream}{self.path}",
                data=body,
                method=self.command,
                headers={"content-type": self.headers.get("content-type", "application/json")},
            )
            try:
                with urlopen(request, timeout=60) as response:  # noqa: S310
                    response_body = response.read()
                    content_type = response.headers.get("content-type", "application/octet-stream")
                    self._send(response.status, response_body, content_type)
            except HTTPError as error:
                self._send(
                    error.code,
                    error.read(),
                    error.headers.get("content-type", "application/json"),
                )
            except URLError:
                self._send(
                    HTTPStatus.BAD_GATEWAY.value,
                    b"Local model server unavailable",
                    "text/plain; charset=utf-8",
                )

        def do_GET(self) -> None:  # noqa: N802
            self._proxy()

        def do_POST(self) -> None:  # noqa: N802
            self._proxy()

        def log_message(self, format: str, *args: object) -> None:
            print("[hERG public proxy] " + format % args, flush=True)

    return AuthenticatedProxy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8789)
    parser.add_argument("--upstream", default="http://127.0.0.1:8788")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password-env", default="HERG_DEMO_PASSWORD")
    return parser


def main() -> int:
    args = _parser().parse_args()
    password = os.environ.get(args.password_env, "")
    if not password:
        raise SystemExit(f"Set the {args.password_env} environment variable")
    server = ThreadingHTTPServer((args.host, args.port), _handler(args.upstream, args.username, password))
    print(f"Authenticated hERG proxy: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
