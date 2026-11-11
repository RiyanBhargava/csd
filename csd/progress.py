"""Stage/reader progress by default; detailed inner loops require CSD_VERBOSE=1."""

import os

from tqdm.auto import tqdm


def verbose():
    return os.environ.get("CSD_VERBOSE") == "1"


def message(value, *, detail=False):
    if not detail or verbose():
        tqdm.write(str(value))


def progress(iterable=None, *, total=None, desc="", leave=False, detail=False):
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=leave,
        dynamic_ncols=True,
        mininterval=1.0,
        disable=os.environ.get("CSD_PROGRESS") == "0" or (detail and not verbose()),
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
