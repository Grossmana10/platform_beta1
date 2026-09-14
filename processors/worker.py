"""Single-consumer Render service; durable leases belong to Platform's D1 queue.

The Site queue endpoints described in README.md must be deployed before this
service is activated. No job URLs, filenames, credentials or audio enter logs.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
from urllib.parse import quote, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

from envelope import MAX_FILE_BYTES, analyze, ffmpeg_binary
from musical import analyze_music


class NoRedirects(HTTPRedirectHandler):
    # Never forward service credentials to a redirected media location.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Client:
    def __init__(self):
        self.origin = os.environ["PLATFORM_ORIGIN"].rstrip("/")
        parsed = urlsplit(self.origin)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.path or parsed.query or parsed.fragment):
            raise ValueError("PLATFORM_ORIGIN must be an HTTPS origin")
        self.secret = os.environ["WAVEFORM_WORKER_SECRET"]
        if len(self.secret) < 32:
            raise ValueError("Worker secret must have at least 32 characters")
        self.dispatch = os.environ.get("SITES_DISPATCH_TOKEN")
        self.opener = build_opener(NoRedirects())

    def request(self, path, payload=None, lease=None):
        headers = {"Authorization": "Bearer " + self.secret}
        if self.dispatch:
            headers["OAI-Sites-Authorization"] = "Bearer " + self.dispatch
        if lease:
            headers["X-Waveform-Lease"] = lease
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        return self.opener.open(Request(self.origin + path, data=data, headers=headers), timeout=45)

    def claim(self):
        with self.request("/api/waveform-jobs/claim", {}) as response:
            if response.status == 204:
                return None
            body = response.read(16385)
            if len(body) > 16384:
                raise ValueError("Invalid claim response")
            job = json.loads(body)
        if (not isinstance(job.get("versionId"), str) or not job["versionId"]
                or len(job["versionId"]) > 200
                or not isinstance(job.get("leaseToken"), str)
                or not 32 <= len(job["leaseToken"]) <= 200):
            raise ValueError("Invalid job lease")
        return job

    def process(self, job):
        route = "/api/waveform-jobs/" + quote(job["versionId"], safe="")
        lease = job["leaseToken"]
        try:
            with tempfile.TemporaryDirectory(prefix="platform-waveform-") as directory:
                path = Path(directory) / "input.audio"
                deadline = time.monotonic() + 180
                with self.request(route + "/media", lease=lease) as response, path.open("wb") as output:
                    advertised = response.headers.get("Content-Length")
                    if advertised is not None and int(advertised) > MAX_FILE_BYTES:
                        raise ValueError("Audio exceeds upload limit")
                    total = 0
                    while True:
                        chunk = response.read(65536)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > MAX_FILE_BYTES or time.monotonic() > deadline:
                            raise ValueError("Audio transfer exceeded its processing budget")
                        output.write(chunk)
                result = analyze(path)
                try:
                    result["music"] = analyze_music(path)
                except Exception:
                    result["music"] = None
            # Originals have already been removed from temporary storage.
            with self.request(route, {"leaseToken": lease, **result}):
                pass
            print("waveform_job_completed", flush=True)
        except Exception:
            # Site decides retry/backoff and ignores a stale lease. Do not
            # submit exception text: upstream errors can contain private data.
            try:
                with self.request(route, {"leaseToken": lease, "error": "analysis_failed"}):
                    pass
            except Exception:
                pass  # Lost acknowledgements are recovered by lease expiry.
            print("waveform_job_failed", flush=True)


def main():
    client = Client()
    ffmpeg_binary()  # Fail startup if the native decoder is unavailable.
    stop = threading.Event()

    def consume():
        while not stop.is_set():
            try:
                job = client.claim()
                if job:
                    client.process(job)
                    continue
            except Exception:
                print("waveform_queue_unavailable", flush=True)
            stop.wait(10)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()

    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            ok = consumer.is_alive() and not stop.is_set()
            body = b'{"status":"running"}' if ok else b'{"status":"stopped"}'
            self.send_response(200 if ok else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "10000"))), Health)
    def shutdown(*_):
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        stop.set()
        consumer.join(timeout=540)


if __name__ == "__main__":
    main()
