#!/usr/bin/env python3
"""Dependency-free Telegram monitor for Slurm jobs.

Credentials are read from ~/.telegram_credentials by default and must define
BOT_TOKEN and CHAT_ID. The script watches Slurm state transitions and optional
log files for crash patterns.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ERROR_RE = re.compile(
    r"Traceback|CUDA out of memory|RuntimeError|ValueError|FAILED|Killed|"
    r"Error:|OSError|ModuleNotFoundError|HTTP [45][0-9][0-9]|segfault",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", nargs="+", help="Slurm job IDs to monitor.")
    parser.add_argument("--label", default="GEPA Slurm monitor")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--credentials", default=str(Path.home() / ".telegram_credentials"))
    parser.add_argument("--state-dir", default=".telegram_monitor_state")
    parser.add_argument("--pid-file", default=None)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--log-glob", action="append", default=[])
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


def send(token: str, chat_id: str, message: str, *, insecure_ssl: bool = False) -> None:
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": message[:3900]}).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=data,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15, context=telegram_ssl_context(insecure_ssl=insecure_ssl)) as response:
        response.read()


def telegram_ssl_context(*, insecure_ssl: bool = False) -> ssl.SSLContext:
    if insecure_ssl or os.environ.get("TELEGRAM_INSECURE_SSL", "").lower() in {"1", "true", "yes"}:
        return ssl._create_unverified_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def safe_send(token: str, chat_id: str, message: str) -> bool:
    try:
        send(token, chat_id, message)
        print(f"telegram send ok: {message.splitlines()[0][:160]}", flush=True)
        return True
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        print(f"telegram send failed: HTTPError {exc.code}: {body}", flush=True)
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError):
            print("telegram send retrying with insecure SSL after certificate verification failure", flush=True)
            try:
                send(token, chat_id, message, insecure_ssl=True)
                print(f"telegram send ok after SSL fallback: {message.splitlines()[0][:160]}", flush=True)
                return True
            except urllib.error.HTTPError as retry_exc:
                body = retry_exc.read().decode("utf-8", errors="replace")[:500]
                print(f"telegram send failed: HTTPError {retry_exc.code}: {body}", flush=True)
            except Exception as retry_exc:
                print(f"telegram send failed after SSL fallback: {type(retry_exc).__name__}: {retry_exc}", flush=True)
            return False
        print(f"telegram send failed: {type(exc).__name__}: {exc}", flush=True)
    except Exception as exc:
        print(f"telegram send failed: {type(exc).__name__}: {exc}", flush=True)
    return False


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip()


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


def squeue_rows(job_id: str) -> list[dict[str, str]]:
    out = run(["squeue", "-j", job_id, "--noheader", "-o", "%i|%T|%j|%M|%R"])
    rows = []
    for line in out.splitlines():
        parts = line.split("|", 4)
        if len(parts) == 5:
            rows.append({"id": parts[0], "state": parts[1], "name": parts[2], "time": parts[3], "reason": parts[4]})
    return rows


def final_state(job_id: str) -> str:
    out = run(["sacct", "-j", job_id, "--format=JobID,JobName,State,ExitCode,Elapsed", "--noheader", "-X"])
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if lines:
        return lines[0]
    out = run(["scontrol", "show", "job", job_id])
    state = re.search(r"JobState=(\S+)", out)
    exit_code = re.search(r"ExitCode=(\S+)", out)
    if state:
        return f"{state.group(1)} ExitCode={exit_code.group(1) if exit_code else 'unknown'}"
    return "unknown final state"


def is_terminal_pending_reason(reason: str) -> bool:
    normalized = reason.replace(" ", "")
    return any(
        marker in normalized
        for marker in (
            "DependencyNeverSatisfied",
            "InvalidAccount",
            "QOSMaxGRESPerUser",
        )
    )


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


def main() -> int:
    args = parse_args()
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    pid_file = Path(args.pid_file) if args.pid_file else state_dir / "telegram_monitor.pid"
    if args.replace:
        replace_previous(pid_file)
    pid_file.write_text(str(os.getpid()), encoding="utf-8")

    token, chat_id = load_credentials(Path(args.credentials))
    safe_send(token, chat_id, f"{args.label}: monitor started\njobs: {', '.join(args.jobs)}")

    previous: dict[str, str] = {}
    offsets: dict[str, int] = {}
    done: set[str] = set()

    while len(done) < len(args.jobs):
        for job_id in args.jobs:
            if job_id in done:
                continue
            rows = squeue_rows(job_id)
            if not rows:
                safe_send(token, chat_id, f"{args.label}: job finished\nid={job_id}\nstate={final_state(job_id)}")
                done.add(job_id)
                continue
            for row in rows:
                old_state = previous.get(row["id"])
                if old_state != row["state"]:
                    safe_send(
                        token,
                        chat_id,
                        f"{args.label}: state change\n"
                        f"id={row['id']}\nname={row['name']}\n"
                        f"{old_state or 'NEW'} -> {row['state']}\n"
                        f"time={row['time']}\nreason={row['reason']}",
                    )
                    previous[row["id"]] = row["state"]
                if row["state"] == "PENDING" and is_terminal_pending_reason(row["reason"]):
                    safe_send(
                        token,
                        chat_id,
                        f"{args.label}: job cannot start\n"
                        f"id={row['id']}\nname={row['name']}\n"
                        f"state={row['state']}\nreason={row['reason']}\n"
                        f"final={final_state(job_id)}",
                    )
                    done.add(job_id)
                    break

        for path in expand_globs(args.log_glob):
            lines, offsets[str(path)] = read_new_lines(path, offsets.get(str(path), 0))
            for raw in lines:
                clean = raw.strip()
                if clean and ERROR_RE.search(clean):
                    safe_send(token, chat_id, f"{args.label}: log alert\nlog={path.name}\n{clean[:2500]}")

        time.sleep(args.poll_seconds)

    safe_send(token, chat_id, f"{args.label}: monitor exiting; all jobs finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
