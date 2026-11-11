import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from csd.budget import atomic_json, exclusive_run, read_budget, supervise


class BudgetTests(unittest.TestCase):
    def test_null_filled_ledger_recovers_without_resetting_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "budget.json"
            atomic_json(ledger, dict(limit_seconds=10, used_seconds=3, status="running"))
            ledger.write_bytes(b"\0" * 83)
            with contextlib.redirect_stdout(io.StringIO()):
                code = supervise([sys.executable, "-B", "-c", "pass"], ledger, 10)
            self.assertEqual(code, 0)
            self.assertGreater(read_budget(ledger)["used_seconds"], 3)
            self.assertEqual(
                json.loads(ledger.read_text()),
                json.loads(ledger.with_suffix(".backup.json").read_text()),
            )

    def test_corrupt_ledger_without_backup_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "budget.json"
            for content in (b"", b"\0" * 83, b"{}", b'{"limit_seconds":10,"used_seconds":NaN}'):
                ledger.write_bytes(content)
                with self.assertRaisesRegex(ValueError, "no valid backup"):
                    supervise([sys.executable, "-B", "-c", "pass"], ledger, 10)
                self.assertEqual(ledger.read_bytes(), content)

    def test_interrupted_replacement_uses_highest_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "budget.json"
            atomic_json(ledger, dict(limit_seconds=10, used_seconds=3, status="running"))
            ledger.with_suffix(".backup.json").write_text(
                json.dumps(dict(limit_seconds=10, used_seconds=5, status="running"))
            )
            self.assertEqual(read_budget(ledger)["used_seconds"], 5)
            ledger.write_text(json.dumps(dict(limit_seconds=11, used_seconds=3)))
            with self.assertRaisesRegex(ValueError, "different limits"):
                read_budget(ledger)

    def test_source_repair_requires_exact_audit_and_preserves_science(self):
        from csd.artifacts import digest, read, save
        from csd.campaign import source_matches

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = {"csd/budget.py": "old", "csd/model.py": "unchanged"}
            files = dict(original, **{"csd/budget.py": "fixed"})
            source = dict(files=files, sha256=digest(files))
            plan = dict(sha256="frozen-plan", source=dict(files=original, sha256=digest(original)))
            with (
                patch("csd.campaign.RESULTS", root),
                patch(
                    "csd.campaign.read",
                    side_effect=lambda name: read(name, directory=root),
                ),
            ):
                self.assertFalse(source_matches("run", plan, source))
                save(
                    "run_source_repair",
                    dict(
                        plan_sha256=plan["sha256"], source=source, changed_files=["csd/budget.py"]
                    ),
                    directory=root,
                )
                self.assertTrue(source_matches("run", plan, source))
                altered = dict(files, **{"csd/model.py": "altered"})
                self.assertFalse(
                    source_matches("run", plan, dict(files=altered, sha256=digest(altered)))
                )

    def test_worker_finishes_and_time_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "budget.json"
            code = supervise([sys.executable, "-B", "-c", "pass"], ledger, 10, heartbeat=0.1)
            value = json.loads(ledger.read_text())
            self.assertEqual(code, 0)
            self.assertGreater(value["used_seconds"], 0)
            self.assertEqual(value["status"], "stage_finished")

    def test_timeout_kills_worker_and_cannot_reset_on_resume(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            ledger = Path(directory) / "budget.json"
            started = time.monotonic()
            code = supervise(
                [sys.executable, "-B", "-c", "import time; time.sleep(60)"],
                ledger,
                0.4,
                heartbeat=0.05,
            )
            self.assertEqual(code, 124)
            self.assertLess(time.monotonic() - started, 5)
            self.assertEqual(json.loads(ledger.read_text())["status"], "budget_exhausted")
            self.assertEqual(supervise([sys.executable, "-B", "-c", "pass"], ledger, 0.4), 124)
            with self.assertRaises(ValueError):
                supervise([sys.executable, "-B", "-c", "pass"], ledger, 20)

    def test_failure_preserves_exit_code_and_remaining_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "budget.json"
            code = supervise([sys.executable, "-B", "-c", "raise SystemExit(3)"], ledger, 10)
            self.assertEqual(code, 3)
            value = json.loads(ledger.read_text())
            self.assertEqual(value["status"], "worker_failed")
            self.assertLess(value["used_seconds"], 10)
            consumed = value["used_seconds"]
            supervise([sys.executable, "-B", "-c", "pass"], ledger, 10)
            self.assertGreater(json.loads(ledger.read_text())["used_seconds"], consumed)

    def test_two_supervisors_cannot_own_one_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.lock"
            with exclusive_run(path):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-c",
                        "from csd.budget import exclusive_run; from pathlib import Path; "
                        f"\nwith exclusive_run(Path({str(path)!r})): pass",
                    ],
                    capture_output=True,
                )
                self.assertNotEqual(result.returncode, 0)
            with exclusive_run(path):
                pass

    def test_timeout_terminates_grandchildren(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            pid_path = root / "grandchild.pid"
            script = (
                "import subprocess,sys,time; from pathlib import Path; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                f"Path({str(pid_path)!r}).write_text(str(child.pid)); time.sleep(60)"
            )
            self.assertEqual(
                supervise(
                    [sys.executable, "-B", "-c", script], root / "budget.json", 1, heartbeat=0.05
                ),
                124,
            )
            self.assertTrue(pid_path.exists(), "Worker did not reach grandchild creation")
            pid = int(pid_path.read_text())
            if sys.platform == "win32":
                import ctypes
                from ctypes import wintypes

                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                kernel.OpenProcess.restype = wintypes.HANDLE
                kernel.GetExitCodeProcess.argtypes = [
                    wintypes.HANDLE,
                    ctypes.POINTER(wintypes.DWORD),
                ]
                kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                handle = kernel.OpenProcess(0x1000, False, pid)
                if handle:
                    code = wintypes.DWORD()
                    kernel.GetExitCodeProcess(handle, ctypes.byref(code))
                    kernel.CloseHandle(handle)
                    self.assertNotEqual(code.value, 259, "Grandchild survived the deadline")
            else:
                import os

                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
