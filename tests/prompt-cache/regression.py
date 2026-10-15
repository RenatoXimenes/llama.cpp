#!/usr/bin/env python3
"""Standalone cached-prompt regression against an isolated llama-server.

Uses synthetic prompts and a caller-supplied GGUF. Never touches worker services
or existing prefill files. All slot files live in a new temporary directory.
"""

import argparse
import concurrent.futures
import json
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


class Server:
    def __init__(self, binary, model, root, context_size=4096, gpu_layers=0, threads=4):
        self.binary, self.model, self.root = binary, model, root
        self.context_size, self.gpu_layers, self.threads = context_size, gpu_layers, threads
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.process = None
        self.log = None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def start(self):
        self.log = (self.root / "server.log").open("ab")
        self.process = subprocess.Popen(
            [str(self.binary), "-m", str(self.model), "--host", "127.0.0.1",
             "--port", str(self.port), "-ngl", str(self.gpu_layers), "-c", str(self.context_size), "-np", "2",
             "-b", "128", "-ub", "128", "-t", str(self.threads), "--cache-ram", "0",
             "--slot-save-path", str(self.root), "--slots"],
            stdout=self.log, stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Server exited; inspect server.log")
            try:
                if self.request("/health", method="GET")[0] == 200:
                    return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(0.2)
        raise TimeoutError("Server did not become ready")

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log:
            self.log.close()

    def request(self, path, body=None, method="POST"):
        raw = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=raw,
                                     headers={"Content-Type": "application/json"}, method=method)
        try:
            with self.opener.open(req, timeout=120) as response:
                data = response.read()
                if response.headers.get("Content-Type", "").startswith("text/event-stream"):
                    return response.status, [json.loads(line[6:]) for line in data.decode().splitlines()
                                             if line.startswith("data: ") and line[6:] != "[DONE]"]
                return response.status, json.loads(data)
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def ok(self, path, body=None, method="POST"):
        status, result = self.request(path, body, method)
        assert status == 200, (path, status, result)
        return result

    def tokens(self, text):
        return self.ok("/tokenize", {"content": text, "add_special": True})["tokens"]

    def completion(self, slot, prompt, **extra):
        return self.ok("/completion", dict(prompt=prompt, id_slot=slot, n_predict=8,
                                           temperature=0, seed=42, cache_prompt=True,
                                           return_tokens=True, **extra))

    def erase(self, slot):
        self.ok(f"/slots/{slot}?action=erase", {})


def run(binary, model, output, context_size=4096, repetitions=25, gpu_layers=0, threads=4):
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="prompt-cache-") as directory:
        root = Path(directory)
        server = Server(binary, model, root, context_size, gpu_layers, threads)
        try:
            server.start()
            for label, prefix in [("alpha", "Once upon a time, Alice found a red flower. " * repetitions),
                                  ("beta", "Once upon a time, Bob found a blue boat. " * repetitions)]:
                prefix_tokens = server.tokens(prefix)
                tail = server.tokens("And then the story continued:")
                server.erase(0)
                initial = server.ok("/completion", dict(prompt=prefix_tokens, id_slot=0,
                                     n_predict=0, cache_prompt=True, temperature=0))
                saved = server.ok("/slots/0?action=save", {"filename": label + ".bin"})
                assert saved["n_saved"] > 0
                server.erase(0)
                restored = server.ok("/slots/0?action=restore", {"filename": label + ".bin"})
                assert restored["n_restored"] == saved["n_saved"]
                continued = server.completion(0, tail, use_slot_prompt=True)
                server.erase(1)
                reference = server.completion(1, prefix_tokens + tail)
                assert continued["content"] == reference["content"], (label, "content mismatch")
                assert continued.get("tokens") == reference.get("tokens")
                assert continued["timings"]["prompt_n"] < reference["timings"]["prompt_n"]
                results.append(dict(test="restore_and_continue_" + label, passed=True,
                                    prefix_tokens=len(prefix_tokens),
                                    full_prompt_n=reference["timings"]["prompt_n"],
                                    cached_prompt_n=continued["timings"]["prompt_n"]))
            # Restore the first document after the second occupied the same slot.
            server.ok("/slots/0?action=restore", {"filename": "alpha.bin"})
            alpha = server.completion(0, tail, use_slot_prompt=True)
            server.erase(1)
            reference = server.completion(1, server.tokens("Once upon a time, Alice found a red flower. " * repetitions) + tail)
            assert alpha["content"] == reference["content"]
            results.append(dict(test="cross_document_isolation", passed=True))
            # Disk state must survive a process restart, rather than just a cache hit.
            server.stop()
            server.start()
            server.ok("/slots/0?action=restore", {"filename": "alpha.bin"})
            after_restart = server.completion(0, tail, use_slot_prompt=True)
            assert after_restart["content"] == alpha["content"]
            results.append(dict(test="restore_after_restart", passed=True))
            server.erase(1)
            status, _ = server.request("/completion", dict(prompt="test", id_slot=1,
                                       use_slot_prompt=True, n_predict=1))
            assert status == 400
            status, _ = server.request("/completion", dict(prompt="test", use_slot_prompt=True,
                                       n_predict=1))
            assert status == 400
            results.append(dict(test="reject_empty_or_unpinned_slot", passed=True))
            # Streaming and independent parallel completions must still work.
            server.erase(0)
            server.erase(1)
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(server.completion, i, "Once upon a time", stream=True)
                           for i in (0, 1)]
                streams = [f.result() for f in futures]
            assert all(isinstance(stream, list) and any(x.get("stop") for x in stream) for stream in streams)
            results.append(dict(test="streaming_parallel_slots", passed=True))
            props = server.ok("/props", method="GET")
            if props.get("chat_template"):
                chat = dict(model="test-model", messages=[{"role": "user", "content": "Say hello."}],
                            max_tokens=8, temperature=0,
                            chat_template_kwargs={"enable_thinking": False})
                response = server.ok("/v1/chat/completions", chat)
                assert response["choices"] and response["usage"]["completion_tokens"] > 0
                stream = server.ok("/v1/chat/completions", dict(chat, stream=True))
                assert any(item.get("choices") for item in stream)
                results.append(dict(test="openai_chat_and_streaming", passed=True))
            (root / "damaged.bin").write_bytes(b"invalid state")
            status, _ = server.request("/slots/0?action=restore", {"filename": "damaged.bin"})
            assert status == 400
            server.ok("/health", method="GET")
            results.append(dict(test="reject_corrupt_state_without_crash", passed=True))
        finally:
            server.stop()
            if (root / "server.log").is_file():
                (output / "server.log").write_bytes((root / "server.log").read_bytes())
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(dict(model=model.name, tests=len(results), results=results)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--prefix-repetitions", type=int, default=25)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.context_size < 128 or args.prefix_repetitions < 1 or args.threads < 1:
        parser.error("Context, prefix repetitions and threads must be positive")
    run(args.binary.resolve(), args.model.resolve(), args.output.resolve(),
        args.context_size, args.prefix_repetitions, args.gpu_layers, args.threads)
