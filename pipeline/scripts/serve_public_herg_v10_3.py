#!/usr/bin/env python3
"""Password-gated reverse proxy for the local hERG V10.3 application."""

from __future__ import annotations

import os
from http.server import ThreadingHTTPServer

import serve_public_herg_v10_1 as proxy


def main() -> int:
    parser = proxy._parser()
    parser.description = __doc__
    parser.set_defaults(
        port=8791,
        upstream="http://127.0.0.1:8792",
        password_env="HERG_V103_DEMO_PASSWORD",
    )
    args = parser.parse_args()
    password = os.environ.get(args.password_env, "")
    if not password:
        raise SystemExit(f"Set the {args.password_env} environment variable")

    proxy.ALLOWED_PATHS = {
        "/",
        "/index.html",
        "/api/info",
        "/api/health",
        "/api/predict",
    }
    server = ThreadingHTTPServer(
        (args.host, args.port),
        proxy._handler(args.upstream, args.username, password),
    )
    print(f"Authenticated hERG V10.3 proxy: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
