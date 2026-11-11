"""Prospective raw-document selection; frozen identities and historical exclusions."""

import hashlib
from pathlib import Path
import re

from .artifacts import ROOT, digest, file_hash, read, save
from .progress import progress

WIKI_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"


def historical_exclusions():
    value = read("exclusions", ROOT / "data")
    return (
        set(value["token_prefix_hashes"]),
        set(value["ids"]),
        dict(ledger_sha256=value["sha256"], prefixes=value["prefixes"]),
    )


def wiki_documents():
    from datasets import load_dataset
    from huggingface_hub import HfApi

    repo = "Salesforce/wikitext"
    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=WIKI_REVISION)
    paths = sorted(
        p for p in files if p.startswith("wikitext-103-raw-v1/train-") and p.endswith(".parquet")
    )
    data = load_dataset(
        "parquet",
        data_files={
            "train": [
                f"https://huggingface.co/datasets/{repo}/resolve/{WIKI_REVISION}/{p}" for p in paths
            ]
        },
        split="train",
        streaming=True,
    )
    current = []
    for row in data:
        text = row["text"]
        if text.startswith(" = ") and not text.startswith(" = ="):
            if current:
                yield "\n".join(current)
            current = [text]
        else:
            current.append(text)
    if current:
        yield "\n".join(current)


def c4_documents():
    from datasets import load_dataset

    url = f"https://huggingface.co/datasets/allenai/c4/resolve/{C4_REVISION}/en/c4-train.00000-of-01024.json.gz"
    for row in load_dataset("json", data_files={"train": url}, split="train", streaming=True):
        yield row["text"]


def normalized_prefix(text):
    return re.sub(r"\s+", " ", text.lower()).strip()[:1024]


def validate_documents(documents):
    seen, prefixes = set(), set()
    for row in documents:
        if row["phase"] not in ("fit", "tune", "development", "confirmation", "long_confirmation"):
            raise ValueError("Invalid data phase")
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["sha256"]:
            raise ValueError("Document digest mismatch")
        prefix = normalized_prefix(row["text"])
        if row["sha256"] in seen or prefix in prefixes:
            raise ValueError("Duplicate or shared-prefix document across partitions")
        seen.add(row["sha256"])
        prefixes.add(prefix)


def prepare(config, name):
    from transformers import AutoTokenizer

    directory = ROOT / "data"
    path = directory / (name + ".json.gz")
    if path.exists():
        result = read(name, directory)
        if result["configuration_sha256"] != digest(config):
            raise ValueError("Dataset name already belongs to a different protocol")
        if result["selection_code_sha256"] != file_hash(__file__):
            raise ValueError("Selection code changed after generating this manifest")
        validate_documents(result["documents"])
        return result
    reference = config["models"][0]
    tokenizer = AutoTokenizer.from_pretrained(reference["id"], revision=reference["revision"])
    blocked_prefix, blocked_ids, excluded_files = historical_exclusions()
    tokenizers = [
        AutoTokenizer.from_pretrained(m["id"], revision=m["revision"]) for m in config["models"]
    ]
    min_lengths = [
        max(m["contexts"]) + max(config["suffix"], config["stream_steps"] + 2)
        for m in config["models"]
    ]
    seen, prefix_seen, documents = set(blocked_ids), set(excluded_files["prefixes"]), []
    for corpus, source in [("wiki", wiki_documents), ("c4", c4_documents)]:
        counts = (
            {"fit": config["fit_articles"], "tune": config["tune_articles"]}
            if corpus == "wiki"
            else {}
        )
        counts.update(
            development=config["development_articles"], confirmation=config["confirmation_articles"]
        )
        phases = [phase for phase, count in counts.items() for _ in range(count)]
        import random

        random.Random(20261025).shuffle(phases)
        selected_count = 0
        with progress(
            total=len(phases), desc=f"Prepare disjoint {corpus} documents", leave=True
        ) as bar:
            for text in source():
                sha = hashlib.sha256(text.encode()).hexdigest()
                prefix = normalized_prefix(text)
                if sha in seen or prefix in prefix_seen:
                    continue
                old_tokens = tokenizer(text, add_special_tokens=False, truncation=False)[
                    "input_ids"
                ]
                if digest(old_tokens[:256]) in blocked_prefix:
                    continue
                lengths = [
                    len(t(text, add_special_tokens=False, truncation=False)["input_ids"])
                    for t in tokenizers
                ]
                if any(
                    length < minimum for length, minimum in zip(lengths, min_lengths, strict=True)
                ):
                    continue
                documents.append(
                    dict(sha256=sha, text=text, corpus=corpus, phase=phases[selected_count])
                )
                seen.add(sha)
                prefix_seen.add(prefix)
                selected_count += 1
                bar.update()
                if selected_count == len(phases):
                    break
        if len([d for d in documents if d["corpus"] == corpus]) != len(phases):
            raise ValueError("Insufficient eligible documents; no partial manifest committed")
    long_requirements = {
        i: max(m["long_contexts"]) + max(config["suffix"], config["stream_steps"] + 2)
        for i, m in enumerate(config["models"])
        if m.get("long_contexts")
    }
    if long_requirements:
        selected_count = 0
        with progress(
            total=config["long_confirmation_articles"],
            desc="Prepare separate long-context WikiText",
            leave=True,
        ) as bar:
            for text in wiki_documents():
                sha = hashlib.sha256(text.encode()).hexdigest()
                prefix = normalized_prefix(text)
                if sha in seen or prefix in prefix_seen:
                    continue
                old_tokens = tokenizer(text, add_special_tokens=False, truncation=False)[
                    "input_ids"
                ]
                if digest(old_tokens[:256]) in blocked_prefix:
                    continue
                if any(
                    len(
                        tokenizers[i](text, add_special_tokens=False, truncation=False)["input_ids"]
                    )
                    < required
                    for i, required in long_requirements.items()
                ):
                    continue
                documents.append(
                    dict(sha256=sha, text=text, corpus="wiki", phase="long_confirmation")
                )
                seen.add(sha)
                prefix_seen.add(prefix)
                selected_count += 1
                bar.update()
                if selected_count == config["long_confirmation_articles"]:
                    break
        if selected_count != config["long_confirmation_articles"]:
            raise ValueError("Insufficient disjoint long articles; no partial manifest saved")
    validate_documents(documents)
    return save(
        name,
        dict(
            configuration_sha256=digest(config),
            documents=documents,
            historical_exclusions=excluded_files,
            dataset_revisions=dict(wikitext=WIKI_REVISION, c4=C4_REVISION),
            final_test_scored=False,
            selection_code_sha256=file_hash(__file__),
            selection_source=Path(__file__).read_text(encoding="utf-8"),
            selection="First eligible disjoint articles in pinned train splits; no model scores",
        ),
        directory,
    )


def tokenize(record, model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"])
    # The original document identity is preserved across tokenizer/backbone changes.
    return [
        dict(
            sha256=d["sha256"],
            corpus=d["corpus"],
            phase=d["phase"],
            tokens=tokenizer(d["text"], add_special_tokens=False, truncation=False)["input_ids"],
        )
        for d in record["documents"]
    ]


def phase(documents, name, corpus=None):
    return [
        d for d in documents if d["phase"] == name and (corpus is None or d["corpus"] == corpus)
    ]
