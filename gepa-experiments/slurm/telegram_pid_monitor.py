#!/usr/bin/env python3
"""Telegram monitor for a long-running local process and its logs.

This complements telegram_monitor.py, which is Slurm-specific. It is used for
direct Docker queues launched outside Slurm, where there is no Slurm job ID to
watch.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ERROR_RE = re.compile(
    r"Traceback|CUDA out of memory|RuntimeError|ValueError|FAILED|Killed|"
    r"Error:|OSError|ModuleNotFoundError|HTTP [45][0-9][0-9]|segfault|"
    r"Critical job failed",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--label", default="GEPA process monitor")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--credentials", default=str(Path.home() / ".telegram_credentials"))
    parser.add_argument("--state-dir", default=".telegram_pid_monitor_state")
    parser.add_argument("--monitor-pid-file", default=None)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--log-glob", action="append", default=[])
    parser.add_argument("--alert-cooldown-seconds", type=int, default=1800)
    parser.add_argument("--max-alerts-per-poll", type=int, default=3)
    return parser.parse_args()


def load_credentials(path: Path) -> tuple[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values["BOT_TOKEN"], values["CHAT_ID"]


def telegram_ssl_context(*, insecure_ssl: bool = False) -> ssl.SSLContext:
    if insecure_ssl or os.environ.get("TELEGRAM_INSECURE_SSL", "").lower() in {"1", "true", "yes"}:
        return ssl._create_unverified_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def send(token: str, chat_id: str, message: str, *, insecure_ssl: bool = False) -> None:
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": message[:3900]}).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=data,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15, context=telegram_ssl_context(insecure_ssl=insecure_ssl)) as response:
        response.read()


def safe_send(token: str, chat_id: str, message: str) -> bool:
    try:
        send(token, chat_id, message)
        print(f"telegram send ok: {message.splitlines()[0][:160]}", flush=True)
        return True
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError):
            try:
                send(token, chat_id, message, insecure_ssl=True)
                print(f"telegram send ok after SSL fallback: {message.splitlines()[0][:160]}", flush=True)
                return True
            except Exception as retry_exc:
                print(f"telegram send failed after SSL fallback: {type(retry_exc).__name__}: {retry_exc}", flush=True)
        else:
            print(f"telegram send failed: {type(exc).__name__}: {exc}", flush=True)
    except Exception as exc:
        print(f"telegram send failed: {type(exc).__name__}: {exc}", flush=True)
    return False


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def replace_previous(pid_file: Path) -> None:
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        return
    if pid == os.getpid() or not is_alive(pid):
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not is_alive(pid):
            return
        time.sleep(0.2)
    if is_alive(pid):
        os.kill(pid, signal.SIGKILL)


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def expand_globs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        if pattern.startswith("/"):
            paths.extend(Path("/").glob(pattern[1:]))
        else:
            paths.extend(Path().glob(pattern))
    return sorted(set(paths))


def read_new_lines(path: Path, offset: int) -> tuple[list[str], int]:
    if not path.exists():
        return [], offset
    size = path.stat().st_size
    if size < offset:
        offset = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        lines = handle.readlines()
        return lines, handle.tell()


def alert_signature(line: str) -> str:
    lowered = line.lower()
    if "connection refused" in lowered:
        return "connection-refused"
    if "internalservererror" in lowered:
        return "litellm-internal-server-error"
    if "apiconnectionerror" in lowered or "connection error" in lowered:
        return "api-connection-error"
    if "cuda out of memory" in lowered or "out of memory" in lowered:
        return "cuda-out-of-memory"
    if "ggml_assert" in lowered or "cudasuccess" in lowered:
        return "llamacpp-cuda-assert"
    if "too many requests" in lowered or "http error 429" in lowered:
        return "rate-limit"
    if "traceback" in lowered:
        return "traceback"
    if "critical job failed" in lowered:
        return "critical-job-failed"
    compact = re.sub(r"\s+", " ", lowered.strip())
    compact = re.sub(r"\b\d+\b", "#", compact)
    return compact[:160]


def main() -> int:
    args = parse_args()
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    monitor_pid_file = Path(args.monitor_pid_file) if args.monitor_pid_file else state_dir / "telegram_pid_monitor.pid"
    if args.replace:
        replace_previous(monitor_pid_file)
    monitor_pid_file.write_text(str(os.getpid()), encoding="utf-8")

    token, chat_id = load_credentials(Path(args.credentials))
    target_pid_file = Path(args.pid_file)
    target_pid = read_pid(target_pid_file)
    safe_send(token, chat_id, f"{args.label}: monitor started\npid_file={target_pid_file}\npid={target_pid}")

    offsets: dict[str, int] = {}
    for path in expand_globs(args.log_glob):
        try:
            offsets[str(path)] = path.stat().st_size
        except OSError:
            pass

    last_alert_at: dict[str, float] = {}
    last_alive: bool | None = None
    while True:
        target_pid = read_pid(target_pid_file)
        alive = target_pid is not None and is_alive(target_pid)
        if alive != last_alive:
            safe_send(token, chat_id, f"{args.label}: process state\npid={target_pid}\nalive={alive}")
            last_alive = alive

        for path in expand_globs(args.log_glob):
            lines, offsets[str(path)] = read_new_lines(path, offsets.get(str(path), 0))
            alerts: dict[str, tuple[str, int]] = {}
            for raw in lines:
                clean = raw.strip()
                if clean and ERROR_RE.search(clean):
                    signature = f"{path.name}:{alert_signature(clean)}"
                    if signature in alerts:
                        first_line, count = alerts[signature]
                        alerts[signature] = (first_line, count + 1)
                    else:
                        alerts[signature] = (clean, 1)

            sent_this_poll = 0
            now = time.time()
            for signature, (first_line, count) in alerts.items():
                previous = last_alert_at.get(signature, 0.0)
                if now - previous < args.alert_cooldown_seconds:
                    print(
                        f"suppressed duplicate alert: signature={signature} count={count}",
                        flush=True,
                    )
                    continue
                if sent_this_poll >= args.max_alerts_per_poll:
                    print(
                        f"suppressed alert due to poll cap: signature={signature} count={count}",
                        flush=True,
                    )
                    continue
                safe_send(
                    token,
                    chat_id,
                    f"{args.label}: log alert\n"
                    f"log={path.name}\n"
                    f"signature={signature}\n"
                    f"matching_lines={count}\n"
                    f"{first_line[:2200]}",
                )
                last_alert_at[signature] = now
                sent_this_poll += 1

        if not alive:
            safe_send(token, chat_id, f"{args.label}: monitor exiting\npid={target_pid}\nalive={alive}")
            return 0

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
