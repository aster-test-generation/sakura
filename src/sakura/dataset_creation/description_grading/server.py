from __future__ import annotations

import json
import secrets
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from .context import DescriptionContextService
from .session import GradingSession
from .web_ui import HTML


class GradingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        session: GradingSession,
        context_service: DescriptionContextService,
        token: str,
    ) -> None:
        super().__init__(address, GradingRequestHandler)
        self.session = session
        self.context_service = context_service
        self.token = token


class GradingRequestHandler(BaseHTTPRequestHandler):
    server: GradingHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._html(HTML)
            return
        if not self._authorized():
            return
        if path == "/api/session":
            self._json(self.server.session.session_payload())
            return
        if path.startswith("/api/entries/"):
            try:
                index = int(path.rsplit("/", 1)[-1])
                payload = self.server.session.entry_payload(index)
                entry = self.server.session.entries[index]
                payload["code_context"] = self.server.context_service.render_entry(entry)
                self._json(payload)
            except (IndexError, ValueError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return
        self._error(HTTPStatus.NOT_FOUND, "Not found")

    def do_PUT(self) -> None:
        path = urlparse(self.path).path
        if not self._authorized() or not self._valid_origin():
            return
        if path.startswith("/api/entries/") and path.endswith("/grades"):
            parts = path.strip("/").split("/")
            try:
                entry_id = int(parts[2])
                self.server.session.set_grades(entry_id, self._read_json())
                self._json(self.server.session.session_payload())
            except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return
        self._error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if not self._authorized() or not self._valid_origin():
            return
        if path == "/api/shutdown":
            self._json({"status": "stopping"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        self._error(HTTPStatus.NOT_FOUND, "Not found")

    def _authorized(self) -> bool:
        if secrets.compare_digest(
            self.headers.get("X-Session-Token", ""), self.server.token
        ):
            return True
        self._error(HTTPStatus.FORBIDDEN, "Invalid session token")
        return False

    def _valid_origin(self) -> bool:
        origin = self.headers.get("Origin")
        expected = f"http://127.0.0.1:{self.server.server_port}"
        if origin is None or origin == expected:
            return True
        self._error(HTTPStatus.FORBIDDEN, "Invalid request origin")
        return False

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 16_384:
            raise ValueError("Request body must contain no more than 16384 bytes")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _html(self, content: str) -> None:
        body = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
            "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status)


def run_description_grader(
    user: str,
    repo_root: Path,
    *,
    open_browser: bool = True,
    reset: bool = False,
) -> None:
    descriptions_dir = repo_root / "outputs" / "descriptions_sample"
    session = GradingSession(user, descriptions_dir, repo_root, reset=reset)
    if session.archived_backup is not None:
        print(f"Archived previous grades to {session.archived_backup}")
    context_service = DescriptionContextService(
        repo_root / "resources" / "datasets",
        repo_root / "resources" / "analysis",
    )
    token = secrets.token_urlsafe(24)
    server = GradingHTTPServer(
        ("127.0.0.1", 0), session, context_service, token
    )
    url = f"http://127.0.0.1:{server.server_port}/?token={token}"
    print(f"Description grader: {url}")
    print(f"Grades: {session.output_path}")
    print("Press Ctrl+C to stop the viewer.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping description grader.")
    finally:
        server.server_close()
