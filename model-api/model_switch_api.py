#!/usr/bin/env python3
"""Single-resident OpenAI chat API for local GGUF and Strata models."""

from __future__ import annotations

import glob
import hmac
import json
import logging
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HOST = os.environ.get("MODEL_API_HOST", "127.0.0.1")
PORT = int(os.environ.get("MODEL_API_PORT", "19080"))
BACKEND_PORT = int(os.environ.get("MODEL_API_BACKEND_PORT", "19081"))
BACKEND = f"http://127.0.0.1:{BACKEND_PORT}"
KEY_FILE = Path(os.environ.get("MODEL_API_KEY_FILE", str(ROOT / "api-key")))
LLAMA = "/home/shiba/.unsloth/llama.cpp/build/bin/llama-server"
K2 = "/home/shiba/multikernel-experiment/research/llama-k2horizon/build-k2-cuda128/bin/llama-server"
MULTIKERNEL_LLAMA = "/home/shiba/llama.cpp/build-main-cuda-rpc-128/bin/llama-server"
STRATA_SERVER = "/home/shiba/multikernel-experiment/research/Strata/serve/server.py"
START_TIMEOUT = 600
MAX_BODY = 32 * 1024 * 1024


def catalog() -> dict[str, dict]:
    entries = json.loads((ROOT / "models.json").read_text())
    result = {}
    for entry in entries:
        item = dict(entry)
        if "pattern" in item:
            matches = sorted(Path(p) for p in glob.glob(item["pattern"]) if Path(p).is_file())
            item["path"] = str(matches[0]) if matches else None
        elif "path" in item:
            item["path"] = str(Path(item["path"])) if (
                Path(item["path"]).is_dir() if item["runner"] == "transformers"
                else Path(item["path"]).is_file()) else None
        else:
            item["path"] = str(ROOT / item["config"])
        item["available"] = item["path"] is not None and (
            (Path(item["path"]) / "model.safetensors").is_file()
            if item["runner"] == "transformers" else Path(item["path"]).is_file())
        if item["id"] in result:
            raise ValueError(f"duplicate model id: {item['id']}")
        result[item["id"]] = item
    return result


class Backend:
    def __init__(self, entries: dict[str, dict]):
        self.entries = entries
        self.lock = threading.RLock()
        self.active: str | None = None
        self.proc: subprocess.Popen | None = None
        self.log = None

    def stop(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                os.killpg(self.proc.pid, signal.SIGTERM)
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=10)
            self.proc = None
        if self.log is not None:
            self.log.close()
            self.log = None
        self.active = None

    def command(self, entry: dict) -> tuple[list[str], dict[str, str]]:
        env = os.environ.copy()
        if entry["runner"] == "transformers":
            env["CUDA_VISIBLE_DEVICES"] = "0"
            return ([entry["python"], str(ROOT / "hf_gpt_backend.py"),
                     "--model", entry["path"], "--alias", entry["id"],
                     "--context", str(entry["context"]), "--host", "127.0.0.1",
                     "--port", str(BACKEND_PORT)], env)
        if entry["runner"] == "strata":
            env["STRATA_DIAG_BYPASS_EXPERT_HITS"] = "1"
            return ([entry.get("python", "/usr/bin/python3"), STRATA_SERVER, "--engine", "strata",
                     "--config", entry["path"], "--host", "127.0.0.1",
                     "--port", str(BACKEND_PORT)], env)
        exe = (K2 if entry["runner"] == "k2" else
               MULTIKERNEL_LLAMA if entry["runner"] == "multikernel" else LLAMA)
        command = [exe, "-m", entry["path"], "--alias", entry["id"],
                   "--host", "127.0.0.1", "--port", str(BACKEND_PORT),
                   "-c", str(entry["context"]), "-ngl", str(entry["ngl"]),
                   "-np", "1", "--jinja", "--no-ui"]
        if entry["runner"] == "multikernel":
            command += ["--rpc", "mkvsock:1:5002", "--device", "CUDA0,RPC0",
                        "--split-mode", "layer", "--tensor-split", entry["tensor_split"],
                        "--fit", "off"]
            env["GGML_MK_VSOCK_NO_READ_SLEEP"] = "1"
            env["GGML_RPC_NO_RDMA"] = "1"
            env.pop("GGML_MK_VSOCK_NO_WRITE_SLEEP", None)
            env.pop("GGML_MK_VSOCK_NO_PROGRESS_SLEEP", None)
        if entry["runner"] != "multikernel":
            if device := entry.get("device"):
                command += ["--device", device]
            if fit := entry.get("fit"):
                command += ["--fit", fit]
        if cache_type := entry.get("cache_type"):
            command += ["--cache-type-k", cache_type, "--cache-type-v", cache_type]
        if flash_attn := entry.get("flash_attn"):
            command += ["--flash-attn", flash_attn]
        if batch_size := entry.get("batch_size"):
            command += ["--batch-size", str(batch_size),
                        "--ubatch-size", str(min(128, batch_size))]
        if entry.get("backend_sampling"):
            command += ["--backend-sampling", "--samplers", "temperature"]
        return command, env

    def ensure(self, model_id: str) -> None:
        entry = self.entries.get(model_id)
        if entry is None:
            raise ValueError(f"unknown model {model_id!r}")
        if not entry["available"]:
            raise ValueError(f"model file unavailable: {model_id}")
        if self.active == model_id and self.proc is not None and self.proc.poll() is None:
            return
        self.stop()
        command, env = self.command(entry)
        log_path = ROOT / "logs" / f"{model_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("ab", buffering=0)
        logging.info("loading %s with %s", model_id, entry["runner"])
        self.proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=self.log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + entry.get("load_timeout", START_TIMEOUT)
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                code = self.proc.returncode
                self.stop()
                raise RuntimeError(f"{model_id} exited during load (code {code}); see {log_path}")
            try:
                with urllib.request.urlopen(f"{BACKEND}/health", timeout=3) as response:
                    if response.status == 200:
                        with urllib.request.urlopen(f"{BACKEND}/v1/models", timeout=3) as models_response:
                            ids = {model.get("id") for model in json.load(models_response).get("data", [])}
                        expected = (json.loads(Path(entry["path"]).read_text())["model_name"]
                                    if entry["runner"] == "strata" else model_id)
                        if expected in ids:
                            self.active = model_id
                            logging.info("ready: %s", model_id)
                            return
            except (OSError, urllib.error.HTTPError):
                pass
            time.sleep(2)
        self.stop()
        raise TimeoutError(f"timed out loading {model_id}; see {log_path}")


ENTRIES = catalog()
MANAGER = Backend(ENTRIES)
API_KEY = KEY_FILE.read_text().strip() if KEY_FILE.is_file() else ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        logging.info("%s %s", self.address_string(), fmt % args)

    def reply(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def authorized(self) -> bool:
        bearer = self.headers.get("Authorization", "")
        supplied = bearer[7:] if bearer.startswith("Bearer ") else self.headers.get("X-API-Key", "")
        return bool(API_KEY) and hmac.compare_digest(supplied, API_KEY)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= MAX_BODY:
            raise ValueError("invalid request size")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_GET(self) -> None:
        if self.path == "/health":
            self.reply(200, {"status": "ok", "active_model": MANAGER.active})
            return
        if not self.authorized():
            self.reply(401, {"error": {"message": "unauthorized"}})
            return
        if self.path == "/v1/models":
            now = int(time.time())
            self.reply(200, {"object": "list", "data": [
                {"id": mid, "object": "model", "created": now, "owned_by": "shiba-local",
                 "metadata": {"active": MANAGER.active == mid, "available": e["available"],
                              "runner": e["runner"], "context_length": e["context"],
                              "note": e.get("note", "")}}
                for mid, e in ENTRIES.items()]})
            return
        self.reply(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        if not self.authorized():
            self.reply(401, {"error": {"message": "unauthorized"}})
            return
        if self.path not in {"/v1/switch", "/v1/unload", "/v1/chat/completions",
                             "/v1/completions", "/v1/messages"}:
            self.reply(404, {"error": {"message": "not found"}})
            return
        try:
            with MANAGER.lock:
                if self.path == "/v1/unload":
                    MANAGER.stop()
                    self.reply(200, {"active_model": None})
                    return
                payload = self.read_json()
                model_id = payload.get("model")
                if not isinstance(model_id, str) or not model_id:
                    raise ValueError("specify a model from GET /v1/models")
                MANAGER.ensure(model_id)
                if self.path == "/v1/switch":
                    self.reply(200, {"active_model": model_id})
                    return
                entry = ENTRIES[model_id]
                if entry["runner"] == "strata" and self.path == "/v1/completions":
                    raise ValueError("Strata supports chat completions, not text completions")
                if entry["runner"] != "strata" and self.path == "/v1/messages":
                    raise ValueError("Anthropic messages are available only for Strata")
                self.forward(payload)
        except (ValueError, json.JSONDecodeError) as exc:
            self.reply(400, {"error": {"message": str(exc)}})
        except TimeoutError as exc:
            self.reply(504, {"error": {"message": str(exc)}})
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            self.reply(503, {"error": {"message": str(exc)}})

    def forward(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(f"{BACKEND}{self.path}", data=body, method="POST",
                                         headers={"Content-Type": "application/json"})
        try:
            upstream = urllib.request.urlopen(request, timeout=600)
        except urllib.error.HTTPError as exc:
            data = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
            self.close_connection = True
            return
        with upstream:
            self.send_response(upstream.status)
            self.send_header("Content-Type", upstream.headers.get("Content-Type", "application/json"))
            self.send_header("Connection", "close")
            if payload.get("stream"):
                self.end_headers()
                for line in upstream:
                    self.wfile.write(line)
                    self.wfile.flush()
            else:
                data = upstream.read()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            self.close_connection = True


def main() -> None:
    if not API_KEY:
        raise SystemExit(f"missing API key file: {KEY_FILE}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    logging.info("model API listening on %s:%d; backend %s; %d models", HOST, PORT, BACKEND, len(ENTRIES))
    try:
        server.serve_forever()
    finally:
        MANAGER.stop()
        server.server_close()


if __name__ == "__main__":
    main()
