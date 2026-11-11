"""Pinned official LongBench prompts and scorer; matched greedy generation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from .artifacts import CACHE, digest, file_hash
from .progress import progress
from .runner import intervention, prefill

OFFICIAL_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
OFFICIAL_BASE = f"https://raw.githubusercontent.com/THUDM/LongBench/{OFFICIAL_COMMIT}/LongBench/"
DATA_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"


def official_files():
    directory = CACHE / "longbench_official" / OFFICIAL_COMMIT
    files = ["eval.py", "metrics.py", "config/dataset2prompt.json", "config/dataset2maxlen.json"]
    for name in files:
        path = directory / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            urllib.request.urlretrieve(OFFICIAL_BASE + name, temporary)
            temporary.replace(path)
    # Execute the pinned authors' scorer, not a locally modified score approximation.
    old_path = list(sys.path)
    previous_metrics = sys.modules.pop("metrics", None)
    try:
        sys.path.insert(0, str(directory))
        spec = importlib.util.spec_from_file_location(
            "csd_official_longbench_eval", directory / "eval.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
        sys.modules.pop("metrics", None)
        if previous_metrics is not None:
            sys.modules["metrics"] = previous_metrics
    return directory, module, {name: file_hash(directory / name) for name in files}


def middle_truncate(tokens, maximum):
    if maximum < 2:
        raise ValueError("Context limit must be at least two")
    return (
        tokens
        if len(tokens) <= maximum
        else tokens[: maximum // 2] + tokens[-(maximum - maximum // 2) :]
    )


def task_assets():
    directory, scorer, hashes = official_files()
    prompts = json.loads((directory / "config/dataset2prompt.json").read_text(encoding="utf-8"))
    limits = json.loads((directory / "config/dataset2maxlen.json").read_text(encoding="utf-8"))
    path = hf_hub_download(
        "zai-org/LongBench", "data.zip", repo_type="dataset", revision=DATA_REVISION
    )
    return (
        scorer,
        prompts,
        limits,
        path,
        dict(
            sources=hashes,
            commit=OFFICIAL_COMMIT,
            dataset_revision=DATA_REVISION,
            data_zip_sha256=file_hash(path),
        ),
    )


@torch.no_grad()
def greedy(model, tokens, transform, maximum, eos, protected):
    generated = []
    with intervention(model, transform, protected) as state:
        cache = prefill(model, tokens[:-1])
        state["enabled"] = True
        current = tokens[-1]
        for _ in range(maximum):
            output = model(
                torch.tensor([[current]], device=model.device),
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            current = int(output.logits[0, -1].argmax())
            if current in eos:
                break
            generated.append(current)
    return generated


def task_samples(path, task, count):
    with zipfile.ZipFile(path) as archive:
        matches = [p for p in archive.namelist() if Path(p).name == task + "_e.jsonl"]
        if len(matches) != 1:
            raise ValueError("Missing official LongBench-E task")
        examples = [
            json.loads(line)
            for line in archive.read(matches[0]).decode("utf-8").split("\n")
            if line.strip()
        ]
    if count < 0 or count > len(examples):
        raise ValueError("Requested more task examples than available")
    # Include all length strata without selecting examples using model scores.
    strata = [
        [
            e
            for e in examples
            if (
                e["length"] < 4000
                if s == 0
                else 4000 <= e["length"] < 8000
                if s == 1
                else e["length"] >= 8000
            )
        ]
        for s in range(3)
    ]
    ordered = []
    for index in range(max(map(len, strata))):
        ordered.extend(group[index] for group in strata if index < len(group))
    return ordered if count == 0 else ordered[:count]


def measure_task(runner, transform, model, task, samples, prompts, limits, protected):
    directory, scorer, _ = official_files()
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"])
    eos = runner.model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    rows = []
    for sample in progress(samples, desc=f"LongBench-E {task}"):
        prompt = prompts[task].format(**sample)
        tokens = middle_truncate(
            tokenizer(prompt, add_special_tokens=False)["input_ids"], max(model["contexts"])
        )
        if task != "triviaqa":
            prompt = tokenizer.decode(tokens)
            tokens = tokenizer.apply_chat_template(
                [dict(role="user", content=prompt)],
                tokenize=True,
                add_generation_prompt=True,
                **model.get("chat_template_kwargs", {}),
            )
            tokens = middle_truncate(tokens, max(model["contexts"]))
        output = greedy(runner.model, tokens, transform, limits[task], eos, protected)
        prediction = tokenizer.decode(output, skip_special_tokens=True)
        metric_prediction = (
            prediction.lstrip("\n").split("\n")[0] if task == "triviaqa" else prediction
        )
        score = max(
            scorer.dataset2metric[task](
                metric_prediction, answer, all_classes=sample.get("all_classes", [])
            )
            for answer in sample["answers"]
        )
        rows.append(
            dict(
                example=str(sample["_id"]),
                document=hashlib.sha256(sample["context"].encode()).hexdigest(),
                prediction=prediction,
                answers=sample["answers"],
                score=float(score),
                prompt_tokens=len(tokens),
                generated_tokens=len(output),
                original_length=sample["length"],
                input_sha256=digest(tokens),
            )
        )
    return rows
