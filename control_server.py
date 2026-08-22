"""Token-protected static file server for RunPod control channels.

RunPod's HTTP proxy makes any exposed port public. This server replaces the
earlier ``python3 -m http.server`` so that inputs, results and logs on the Pod
are only readable with the per-run bearer token that the controller generated
and passed through the Pod environment (``CONTROL_TOKEN``). GET/HEAD only, no
directory listing, constant-time token comparison.
"""

from __future__ import annotations

import argparse
import hmac
import http.server
import os
import sys
from pathlib import Path


class TokenProtectedHandler(http.server.SimpleHTTPRequestHandler):
    token: str = ""

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.token}"
        return bool(self.token) and hmac.compare_digest(header, expected)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if not self._authorized():
            self.send_error(401, "Unauthorized")
            return
        if self.path.endswith("/"):
            self.send_error(404, "Not Found")
            return
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        if not self._authorized():
            self.send_error(401, "Unauthorized")
            return
        super().do_HEAD()

    def list_directory(self, path):  # type: ignore[override]
        self.send_error(404, "Not Found")
        return None

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        sys.stderr.write("%s - - %s\n" % (self.address_string(), format % args))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token-env", default="CONTROL_TOKEN")
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env, "")
    if len(token) < 32:
        print(f"{args.token_env} must hold a token of at least 32 characters", file=sys.stderr)
        return 2
    handler = type(
        "BoundHandler",
        (TokenProtectedHandler,),
        {"token": token},
    )
    directory = str(args.directory.resolve())

    def factory(*handler_args, **handler_kwargs):
        return handler(*handler_args, directory=directory, **handler_kwargs)

    with http.server.ThreadingHTTPServer(("0.0.0.0", args.port), factory) as server:
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
