"""Atomic, immutable, content-verified experiment and checkpoint records."""

from contextlib import contextmanager
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(
    os.environ.get(
        "CSD_CACHE_DIR", str(Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "csd")
    )
)
RESULTS = ROOT / "results"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot():
    files = {
        p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(
            [
                *(
                    p
                    for p in (ROOT / "csd").rglob("*")
                    if p.is_file()
                    and (
                        p.suffix in (".py", ".json")
                        or p.name.endswith("LICENSE")
                        or p.name == "NOTICE"
                    )
                ),
                *(ROOT / "configs").glob("*.json"),
                *(ROOT / "tests").rglob("*.py"),
                ROOT / "requirements.txt",
                ROOT / "data" / "exclusions.json.gz",
            ]
        )
        if p.suffix != ".gz"
    }
    files["data/exclusions.json.gz:sha256"] = file_hash(ROOT / "data" / "exclusions.json.gz")
    return dict(sha256=digest(files), files=files)


def valid_name(name):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}", name):
        raise ValueError("Invalid run name")
    return name


def read(name, directory=RESULTS):
    path = Path(directory) / (valid_name(name) + ".json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        value = json.load(f)
    sha = value.pop("sha256")
    if digest(value) != sha:
        raise ValueError(f"Corrupted record: {path}")
    return dict(value, sha256=sha)


@contextmanager
def exclusive_lock(path):
    """OS locks release on crashes; an abandoned lock file never blocks a resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("This resource is already locked by another process") from error
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def save(name, value, directory=RESULTS):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (valid_name(name) + ".json.gz")
    lock = directory / (name + ".lock")
    with exclusive_lock(lock):
        temporary = None
        try:
            if path.exists():
                raise FileExistsError(path)
            value = {k: v for k, v in value.items() if k != "sha256"}
            value["sha256"] = digest(value)
            fd, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
            os.close(fd)
            with gzip.open(temporary, "wt", encoding="utf-8") as f:
                json.dump(value, f, allow_nan=False, separators=(",", ":"))
            os.replace(temporary, path)
            return value
        finally:
            if temporary and Path(temporary).exists():
                Path(temporary).unlink()


def environment():
    import torch

    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        packages={
            p: importlib.metadata.version(p)
            for p in [
                "torch",
                "numpy",
                "scipy",
                "transformers",
                "datasets",
                "huggingface-hub",
                "tqdm",
                "rouge",
                "jieba",
                "fuzzywuzzy",
                "hqq",
                "einops",
                "accelerate",
                "termcolor",
            ]
        },
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cuda=torch.version.cuda,
        deterministic=torch.are_deterministic_algorithms_enabled(),
    )


def checkpoint(path, state):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = torch.load(path, weights_only=True, map_location="cpu")
        if previous.keys() != state.keys() or any(
            not torch.equal(previous[k], state[k].cpu()) for k in state
        ):
            raise FileExistsError("Existing orphan checkpoint differs; refusing to overwrite it")
        return dict(path=str(path.resolve()), sha256=file_hash(path))
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
    return dict(path=str(path.resolve()), sha256=file_hash(path))


def load_checkpoint(record, device="cuda"):
    import torch

    if file_hash(record["path"]) != record["sha256"]:
        raise ValueError("Checkpoint digest mismatch")
    return torch.load(record["path"], weights_only=True, map_location=device)
