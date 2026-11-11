"""A cumulative wall-clock cap; the experiment worker cannot silently run for days."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .artifacts import ROOT, valid_name, exclusive_lock

exclusive_run = exclusive_lock


def atomic_json(path, value):
    """Flush a redundant reservation before replacing the main ledger."""
    encoded = json.dumps(value, indent=2, allow_nan=False).encode("utf-8")
    for target in (path.with_suffix(".backup.json"), path):
        temporary = target.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)


def read_budget(path):
    """Recover the highest valid reservation; never reset a corrupt ledger."""
    candidates, invalid = [], []
    for candidate in (path, path.with_suffix(".backup.json")):
        if not candidate.exists():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
            limit, used = value["limit_seconds"], value["used_seconds"]
            if (
                isinstance(limit, bool)
                or isinstance(used, bool)
                or not math.isfinite(limit)
                or not math.isfinite(used)
                or not 0 <= used <= limit
                or limit <= 0
            ):
                raise ValueError("Invalid time reservation")
            candidates.append(value)
        except (ValueError, KeyError, TypeError):
            invalid.append(candidate)
    if not candidates:
        if invalid:
            raise ValueError(
                f"Corrupt budget ledger and no valid backup: {path}. "
                "Preserve the files and recover prior usage; do not reset the budget."
            )
        return {}
    if len({v["limit_seconds"] for v in candidates}) != 1:
        raise ValueError("Budget ledger and backup have different limits")
    if invalid:
        print("Recovering budget from the valid redundant reservation.", flush=True)
    return max(candidates, key=lambda v: v["used_seconds"])


class ProcessTree:
    """Kill the entire owned worker tree, including if its supervisor is interrupted."""

    def __init__(self, command):
        self.job = None
        self.process = None
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class BasicLimits(ctypes.Structure):
                _fields_ = [
                    ("process_time", ctypes.c_int64),
                    ("job_time", ctypes.c_int64),
                    ("flags", wintypes.DWORD),
                    ("minimum", ctypes.c_size_t),
                    ("maximum", ctypes.c_size_t),
                    ("active", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t),
                    ("priority", wintypes.DWORD),
                    ("scheduling", wintypes.DWORD),
                ]

            class IoCounters(ctypes.Structure):
                _fields_ = [
                    (name, ctypes.c_uint64)
                    for name in (
                        "read_ops",
                        "write_ops",
                        "other_ops",
                        "read_bytes",
                        "write_bytes",
                        "other_bytes",
                    )
                ]

            class ExtendedLimits(ctypes.Structure):
                _fields_ = [
                    ("basic", BasicLimits),
                    ("io", IoCounters),
                    ("process_memory", ctypes.c_size_t),
                    ("job_memory", ctypes.c_size_t),
                    ("peak_process", ctypes.c_size_t),
                    ("peak_job", ctypes.c_size_t),
                ]

            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
            self.kernel.SetInformationJobObject.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
            ]
            self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.job = self.kernel.CreateJobObjectW(None, None)
            limits = ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self.job or not self.kernel.SetInformationJobObject(
                self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
            ):
                self.close()
                raise ctypes.WinError(ctypes.get_last_error())
            self.process = subprocess.Popen(command, cwd=ROOT)
            if not self.kernel.AssignProcessToJobObject(self.job, int(self.process._handle)):
                self.process.kill()
                self.process.wait()
                self.close()
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            self.process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)

    def close(self):
        if self.job:
            self.kernel.CloseHandle(self.job)
            self.job = None
        elif os.name != "nt" and self.process is not None and self.process.poll() is None:
            import signal

            os.killpg(self.process.pid, signal.SIGKILL)
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()


def supervise(command, ledger_path, limit_seconds, *, heartbeat=5.0):
    """Persist charged time before each wait; crashes cannot reset the budget."""
    ledger_path = Path(ledger_path)
    with exclusive_run(ledger_path.with_suffix(".lock")):
        old = read_budget(ledger_path)
        if old and old["limit_seconds"] != limit_seconds:
            raise ValueError("A run's cumulative time budget cannot be changed")
        used = float(old.get("used_seconds", 0))
        remaining = max(0, limit_seconds - used)
        if remaining <= 0:
            print(
                "Five-hour budget already exhausted; completed artifacts are preserved.", flush=True
            )
            return 124
        started = time.monotonic()
        worker = None
        status = "interrupted"
        try:
            # Reserve an interval before starting. A killed supervisor overcharges
            # at most one heartbeat instead of losing consumed GPU time.
            atomic_json(
                ledger_path,
                dict(
                    limit_seconds=limit_seconds,
                    used_seconds=min(limit_seconds, used + heartbeat),
                    status="running",
                ),
            )
            worker = ProcessTree(command)
            while True:
                left = remaining - (time.monotonic() - started)
                if left <= 0:
                    status = "budget_exhausted"
                    print(
                        "Five-hour budget reached. Stopping owned workers; partial runs remain incomplete.",
                        flush=True,
                    )
                    return 124
                wait = min(heartbeat, left)
                atomic_json(
                    ledger_path,
                    dict(
                        limit_seconds=limit_seconds,
                        used_seconds=min(limit_seconds, used + time.monotonic() - started + wait),
                        status="running",
                    ),
                )
                try:
                    code = worker.process.wait(timeout=wait)
                    status = "stage_finished" if code == 0 else "worker_failed"
                    return code
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if worker is not None:
                worker.close()
            consumed = min(limit_seconds, used + time.monotonic() - started)
            atomic_json(
                ledger_path, dict(limit_seconds=limit_seconds, used_seconds=consumed, status=status)
            )


def run_campaign(name, config_path, stage):
    valid_name(name)
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    hours = config["max_wall_hours"]
    if not 0 < hours <= 5:
        raise ValueError("The campaign budget must be at most five hours")
    # Leave a small margin for process teardown within the requested five hours.
    limit = hours * 3600 - 10
    print(
        f"CSD budgeted study: cumulative limit {hours:g} hours, including preparation.", flush=True
    )
    command = [sys.executable, "-u", "-B", "-m", "csd.campaign", name, str(path), stage]
    code = supervise(command, ROOT / "results" / (name + "_budget.json"), limit)
    if code == 124:
        print(
            f"Use python -B -m csd report --name {name} to inspect completed/missing panels.",
            flush=True,
        )
    raise SystemExit(code)
