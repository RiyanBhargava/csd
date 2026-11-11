"""Frozen, resumable calibration and evaluation for the single CSD architecture."""

import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from .artifacts import (
    CACHE,
    RESULTS,
    checkpoint,
    environment,
    load_checkpoint,
    read,
    save,
    snapshot,
    valid_name,
)
from .baselines import Eden, Hqq, provenance
from .calibration import fit_decoder
from .data import phase, prepare, tokenize
from .evaluation import benchmark, mechanism, payload, paired_summary
from .model import TRAINED, CooperativeDecoder, CorrectionDecoder, DiagonalDecoder, NeuralDecoder
from .priors import fit_structured_prior
from .progress import message, progress
from .runner import Runner, stream
from .tasks import measure_task, task_assets, task_samples

READERS = (
    "bf16",
    "eden_unbiased",
    "drive",
    "matched_norm",
    "hqq1_key",
    *TRAINED,
)

BUDGET_REQUIRED = {
    "bf16",
    "eden_unbiased",
    "drive",
    "hqq1_key",
    "diagonal",
    "continued_diagonal",
    "gaussian",
    "csd",
    "headwise",
    "layerwise",
}


def trained_methods(config):
    return tuple(method for method in TRAINED if method in config["methods"])


def checkpoint_methods(config):
    return trained_methods(config)


def configure():
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def validate_config(config):
    if config["seeds"] != [11, 22, 33]:
        raise ValueError("The frozen protocol requires seeds 11, 22, 33")
    selected = set(config["methods"])
    if (
        not BUDGET_REQUIRED.issubset(selected)
        or not selected.issubset(READERS)
        or len(selected) != len(config["methods"])
    ):
        raise ValueError("Canonical reader coverage mismatch")
    if not 0 < config["max_wall_hours"] <= 5:
        raise ValueError("The experiment budget cannot exceed five hours")
    if config["group_size"] != 3 or config["rank_per_layer"] != 16:
        raise ValueError("Changing source layout creates a different architecture")
    if config["epochs"] < 1 or min(config["lrs"]) <= 0:
        raise ValueError("Invalid calibration budget")
    if not set(config["lrs"]).issubset(config["linear_lrs"]):
        raise ValueError("Linear control must receive at least the common LR grid")
    if len(config["models"]) != 1 or len({m["name"] for m in config["models"]}) != len(
        config["models"]
    ):
        raise ValueError("The budgeted study uses one backbone")
    if min(config["train_cuts"]) <= config["protected_tokens"] + 1:
        raise ValueError("Calibration must contain compressed tokens")
    if not set(config["diagnostic_methods"]).issubset(READERS):
        raise ValueError("Unknown diagnostic reader")
    for key in (
        "diagnostic_methods",
        "task_methods",
        "mechanism_methods",
        "efficiency_methods",
        "stream_methods",
    ):
        if not set(config[key]).issubset(selected):
            raise ValueError("A panel references a reader outside the selected study")
    if (
        not set(config["task_methods"]).issubset(READERS)
        or config["evaluation_batch_documents"] < 1
    ):
        raise ValueError("Invalid task coverage or document batch size")
    primary = config["primary"]
    if (
        primary["model"] not in {m["name"] for m in config["models"]}
        or primary["comparator"] != "gaussian"
        or primary["metric"] != "nll"
        or primary["minimum_gain"] != 0.01
    ):
        raise ValueError("Prespecified primary objective changed")
    for model in config["models"]:
        if len(model["revision"]) != 40 or max(config["tune_cuts"]) > max(model["contexts"]):
            raise ValueError("Pinned model revision and supported tune length required")
    return config


def required_panels(config, model):
    result = []
    for context in model["contexts"]:
        result.append(
            dict(
                kind="continuation",
                phase="confirmation",
                context=context,
                methods=config["methods"],
            )
        )
    for context in model.get("long_contexts", []):
        result.append(
            dict(
                kind="continuation",
                phase="long_confirmation",
                context=context,
                methods=config["methods"],
            )
        )
    for kind in ("stream", "mechanism", "efficiency"):
        result.append(
            dict(
                kind=kind,
                phase="confirmation",
                context=max(model["contexts"]),
                methods=config.get(kind + "_methods", config["diagnostic_methods"]),
            )
        )
    for recent in config["recent_windows"]:
        if recent != config["protected_tokens"]:
            result.append(
                dict(
                    kind="recent",
                    phase="confirmation",
                    recent=recent,
                    context=max(model["contexts"]),
                    methods=config["diagnostic_methods"],
                )
            )
    for task in config["tasks"]:
        result.append(
            dict(
                kind="task",
                task=task,
                phase="confirmation",
                context=max(model["contexts"]),
                methods=config["task_methods"],
            )
        )
    return result


def existing(name, plan):
    if not (RESULTS / (name + ".json.gz")).exists():
        return None
    record = read(name)
    if record["plan_sha256"] != plan["sha256"]:
        raise ValueError("Run name belongs to another frozen plan; use a new run name")
    return record


def commit(name, record, plan):
    return save(name, dict(**record, plan_sha256=plan["sha256"]))


def source_matches(name, plan, source):
    """Permit only an explicitly audited, exact supervisor repair snapshot."""
    if plan["source"]["sha256"] == source["sha256"]:
        return True
    identifier = name + "_source_repair"
    if not (RESULTS / (identifier + ".json.gz")).exists():
        return False
    repair = read(identifier)
    changed = {
        k
        for k in plan["source"]["files"].keys() | source["files"].keys()
        if plan["source"]["files"].get(k) != source["files"].get(k)
    }
    return (
        repair["plan_sha256"] == plan["sha256"]
        and repair["source"] == source
        and changed == set(repair["changed_files"])
        and changed <= {"csd/budget.py", "csd/campaign.py", "tests/test_budget.py"}
    )


def plan_run(name, config_path):
    valid_name(name)
    if (RESULTS / (name + "_report.json")).exists() and not (
        RESULTS / (name + "_plan.json.gz")
    ).exists():
        raise ValueError("Historical result export exists for this name. Use a new campaign name.")
    config = validate_config(json.loads(Path(config_path).read_text(encoding="utf-8")))
    source = snapshot()
    plan_name = name + "_plan"
    if (RESULTS / (plan_name + ".json.gz")).exists():
        plan = read(plan_name)
        if not source_matches(name, plan, source) or plan["config"] != config:
            raise ValueError(
                "Source or config changed after freezing; refusing mixed-source resume"
            )
        current = environment()
        if (
            any(plan["environment"][k] != current[k] for k in ("packages", "gpu", "cuda"))
            or plan["baselines"] != provenance()
        ):
            raise ValueError(
                "Environment/baseline identity changed; refusing mixed-environment resume"
            )
        return plan
    return save(
        plan_name,
        dict(
            config=config,
            source=source,
            environment=environment(),
            baselines=provenance(),
            task="Five-hour, single-backbone confirmation of frozen CSD; no architecture search",
            prior_screening_gate_passed=False,
            claims=dict(
                novelty_guaranteed=False, publication_ready=False, packed_serving_speedup=False
            ),
        ),
    )


def model_state(path, state):
    return checkpoint(path, {k: v.detach().cpu() for k, v in state.items()})


def prepare_prior(runner, documents, config, name, model, plan):
    record_name = f"{name}_{model}_prior"
    previous = existing(record_name, plan)
    if previous:
        return load_checkpoint(previous["checkpoint"])
    fitting = phase(documents, "fit")
    cut = max(config["train_cuts"])
    with torch.no_grad():
        centers = torch.stack(
            [
                sum(
                    runner.data(a, cut)["keys"][layer][0].float().mean(-2, keepdim=True)
                    for a in fitting
                )
                / len(fitting)
                for layer in range(runner.layers)
            ]
        )
    buffers, diagnostics = fit_structured_prior(
        runner, fitting, cut, config["group_size"], config["rank_per_layer"]
    )
    buffers["centers"] = centers
    record = commit(
        record_name,
        dict(
            checkpoint=model_state(CACHE / name / model / "prior.pt", buffers),
            fit_ids=[a["sha256"] for a in fitting],
            cross_fit="Leave-one-fitting-document-out",
            diagnostics=diagnostics,
            identity=runner.identity,
        ),
        plan,
    )
    return load_checkpoint(record["checkpoint"])


def make(method, prior, diagonal, seed, theta, config):
    if method not in READERS:
        raise ValueError(f"Unknown reader: {method}")
    if method == "bf16":
        return None
    if method == "hqq1_key":
        return Hqq()
    if method in ("eden_unbiased", "drive", "matched_norm"):
        modes = dict(eden_unbiased="unbiased", drive="mse", matched_norm="matched_norm")
        return Eden(prior["centers"], seed, modes[method])
    if method in ("diagonal", "continued_diagonal"):
        decoder = DiagonalDecoder(prior["centers"], seed, theta)
        if diagonal is not None:
            decoder.diagonal.data.copy_(diagonal)
        return decoder
    if method in ("signed", "linear"):
        return CorrectionDecoder(prior["centers"], diagonal, seed, theta, method)
    if method == "neural":
        return NeuralDecoder(prior["centers"], diagonal, seed, theta)
    return CooperativeDecoder(
        prior,
        diagonal,
        seed,
        theta,
        method,
        group_size=config["group_size"],
        token_chunk=config["token_chunk"],
    )


def calibrate(runner, documents, config, name, model, plan):
    prior = prepare_prior(runner, documents, config, name, model, plan)
    phases = {p: phase(documents, p) for p in ("fit", "tune")}
    for seed in config["seeds"]:
        anchor = None
        for method in progress(
            trained_methods(config), desc=f"{model} seed {seed}: calibration", leave=True
        ):
            record_name = f"{name}_{model}_{method}_{seed}"
            record = existing(record_name, plan)
            if record is None:
                choices, events, best = [], [], None
                rates = config["linear_lrs"] if method == "linear" else config["lrs"]
                started = time.perf_counter()
                for lr in rates:
                    torch.manual_seed(seed)
                    decoder = make(method, prior, anchor, seed, runner.theta, config)
                    state, choice, history = fit_decoder(
                        decoder,
                        runner,
                        phases,
                        config["train_cuts"],
                        config["tune_cuts"],
                        seed,
                        lr,
                        config["epochs"],
                        method,
                    )
                    choices.append(choice)
                    events.extend(history)
                    if best is None or choice["tune_nll"] < best[0]["tune_nll"]:
                        best = choice, state
                    del decoder
                record = commit(
                    record_name,
                    dict(
                        method=method,
                        seed=seed,
                        selected=best[0],
                        trials=choices,
                        events=events,
                        checkpoint=model_state(
                            CACHE / name / model / f"{method}_{seed}.pt", best[1]
                        ),
                        prior_sha256=existing(f"{name}_{model}_prior", plan)["sha256"],
                        fit_ids=[a["sha256"] for a in phases["fit"]],
                        tune_ids=[a["sha256"] for a in phases["tune"]],
                        wall_seconds=time.perf_counter() - started,
                    ),
                    plan,
                )
            if method == "diagonal":
                anchor = load_checkpoint(record["checkpoint"])["diagonal"]
    # The marker is written only after every calibration checkpoint exists.
    marker = f"{name}_{model}_calibrated"
    if existing(marker, plan) is None:
        commit(
            marker,
            dict(
                checkpoints=[
                    existing(f"{name}_{model}_{method}_{seed}", plan)["sha256"]
                    for seed in config["seeds"]
                    for method in checkpoint_methods(config)
                ]
            ),
            plan,
        )
    return prior


def restore(method, runner, prior, config, name, model, seed, plan):
    diagonal = load_checkpoint(existing(f"{name}_{model}_diagonal_{seed}", plan)["checkpoint"])[
        "diagonal"
    ]
    decoder = make(method, prior, diagonal, seed, runner.theta, config)
    if method in TRAINED:
        decoder.load_state_dict(
            load_checkpoint(existing(f"{name}_{model}_{method}_{seed}", plan)["checkpoint"])
        )
    if hasattr(decoder, "eval"):
        decoder.eval()
    return decoder


def panel_name(name, model, panel, method, seed):
    detail = panel.get("task", str(panel["context"]))
    if "recent" in panel:
        detail += "_" + str(panel["recent"])
    return f"{name}_{model}_{panel['kind']}_{panel['phase']}_{detail}_{method}_{seed}"


def continuation_panel(runner, documents, config, name, model, plan, prior, panel):
    """Document batches reuse bounded teacher caches across all readers and seeds."""
    pending = [
        (method, seed)
        for seed in config["seeds"]
        for method in panel["methods"]
        if existing(panel_name(name, model, panel, method, seed), plan) is None
    ]
    if not pending:
        return
    size = config["evaluation_batch_documents"]
    for first in progress(range(0, len(documents), size), desc="Document batches"):
        chunk = documents[first : first + size]
        for method, seed in pending:
            base = panel_name(name, model, panel, method, seed)
            fragment = base + f"_part{first // size}"
            if existing(fragment, plan):
                continue
            seed0 = config["seeds"][0]
            if method in ("bf16", "hqq1_key") and seed != seed0:
                source = existing(
                    panel_name(name, model, panel, method, seed0) + f"_part{first // size}", plan
                )
                if source is None:
                    # A fully completed reference may have been resumed without fragments.
                    source = existing(panel_name(name, model, panel, method, seed0), plan)
                if source:
                    ids = {a["sha256"] for a in chunk}
                    commit(
                        fragment,
                        dict(
                            method=method,
                            seed=seed,
                            panel=panel,
                            reused_deterministic_record=source["sha256"],
                            observations=[r for r in source["observations"] if r["article"] in ids],
                        ),
                        plan,
                    )
                    continue
            decoder = restore(method, runner, prior, config, name, model, seed, plan)
            started = time.perf_counter()
            observations = runner.measure(chunk, [panel["context"]], decoder)
            commit(
                fragment,
                dict(
                    method=method,
                    seed=seed,
                    panel=panel,
                    observations=observations,
                    identity=runner.identity,
                    wall_seconds=time.perf_counter() - started,
                ),
                plan,
            )
            del decoder
    for method, seed in pending:
        base = panel_name(name, model, panel, method, seed)
        parts = [
            existing(base + f"_part{first // size}", plan)
            for first in range(0, len(documents), size)
        ]
        observations = [r for part in parts for r in part["observations"]]
        if [r["article"] for r in observations] != [d["sha256"] for d in documents]:
            raise ValueError("Incomplete document fragments; no final reader record committed")
        commit(
            base,
            dict(
                method=method,
                seed=seed,
                panel=panel,
                observations=observations,
                fragments=[p["sha256"] for p in parts],
                identity=runner.identity,
                wall_seconds=sum(p.get("wall_seconds", 0) for p in parts),
            ),
            plan,
        )


def evaluate(runner, documents, config, name, model, plan, prior):
    if not existing(f"{name}_{model['name']}_calibrated", plan):
        raise ValueError("All calibration selections must be frozen before test evaluation")
    assets = None
    for panel in progress(
        required_panels(config, model), desc=f"{model['name']}: panels", leave=True
    ):
        docs = phase(documents, panel["phase"])
        if panel["kind"] not in ("continuation", "task"):
            docs = [
                a
                for corpus in ("wiki", "c4")
                for a in phase(docs, panel["phase"], corpus)[: config["diagnostic_articles"]]
            ]
        runner.protected = panel.get("recent", config["protected_tokens"])
        if panel["kind"] in ("continuation", "recent"):
            continuation_panel(runner, docs, config, name, model["name"], plan, prior, panel)
            runner.protected = config["protected_tokens"]
            continue
        for seed in config["seeds"]:
            for method in progress(panel["methods"], desc=f"{panel['kind']} seed {seed}"):
                record_name = panel_name(name, model["name"], panel, method, seed)
                if existing(record_name, plan):
                    continue
                # Deterministic published references do not acquire artificial seed variability.
                if method in ("bf16", "hqq1_key") and seed != config["seeds"][0]:
                    first = existing(
                        panel_name(name, model["name"], panel, method, config["seeds"][0]), plan
                    )
                    if first:
                        commit(
                            record_name,
                            dict(
                                method=method,
                                seed=seed,
                                panel=panel,
                                reused_deterministic_record=first["sha256"],
                                observations=first["observations"],
                            ),
                            plan,
                        )
                        continue
                decoder = restore(method, runner, prior, config, name, model["name"], seed, plan)
                started = time.perf_counter()
                kind, cut = panel["kind"], panel["context"]
                if kind == "stream":
                    observations = []
                    for article in progress(docs, desc="Sequential articles", detail=True):
                        # Keep log distributions only in bounded CPU memory during one document.
                        _, _, teacher = stream(
                            runner.model,
                            article["tokens"],
                            cut,
                            config["stream_steps"],
                            None,
                            runner.protected,
                        )
                        losses, divergences, _ = stream(
                            runner.model,
                            article["tokens"],
                            cut,
                            config["stream_steps"],
                            decoder,
                            runner.protected,
                            teacher,
                        )
                        observations.append(
                            dict(
                                article=article["sha256"],
                                corpus=article["corpus"],
                                nll=float(np.mean(losses)),
                                kl=float(np.mean(divergences)),
                                token_nll=losses,
                                token_kl=divergences,
                            )
                        )
                elif kind == "mechanism":
                    observations = [mechanism(runner, decoder, a, cut) for a in docs]
                elif kind == "efficiency":
                    observations = [
                        dict(
                            **benchmark(
                                runner,
                                decoder,
                                docs[0],
                                cut,
                                warmup=config["benchmark_warmup"],
                                repetitions=config["benchmark_repetitions"],
                            ),
                            payload=payload(runner, decoder, cut - 1),
                        )
                    ]
                elif kind == "task":
                    if assets is None:
                        assets = task_assets()
                    _, prompts, limits, path, identity = assets
                    observations = measure_task(
                        runner,
                        decoder,
                        model,
                        panel["task"],
                        task_samples(path, panel["task"], config["task_articles"]),
                        prompts,
                        limits,
                        runner.protected,
                    )
                else:
                    raise ValueError("Unknown experiment panel")
                record = dict(
                    method=method,
                    seed=seed,
                    panel=panel,
                    observations=observations,
                    wall_seconds=time.perf_counter() - started,
                    identity=runner.identity,
                )
                if kind == "task":
                    record["task_source"] = identity
                commit(record_name, record, plan)
                del decoder
        runner.protected = config["protected_tokens"]


def run(name, config_path, stage="all", dry_run=False):
    configure()
    config = validate_config(json.loads(Path(config_path).read_text(encoding="utf-8")))
    if dry_run:
        for model in config["models"]:
            panels = required_panels(config, model)
            count = sum(len(p["methods"]) * len(config["seeds"]) for p in panels)
            message(
                f"{model['name']}: {len(trained_methods(config))} calibrated readers, "
                f"{len(panels)} evaluation panels, {count} reader records"
            )
        message(
            f"Seeds {config['seeds']}; {config['epochs']} epochs/LR; "
            f"{len(config['methods'])} selected readers; {config['max_wall_hours']:g}-hour cumulative cap. "
            "No model loaded; no test scores read."
        )
        return
    plan = plan_run(name, config_path)
    manifest = prepare(config, name)
    for model in config["models"]:
        message(f"{model['label']}: {stage}")
        documents = tokenize(manifest, model)
        runner = Runner(
            model["id"],
            model["revision"],
            suffix=config["suffix"],
            protected=config["protected_tokens"],
            cache_windows=config["cache_windows"],
            reference_cache_gib=config["reference_cache_gib"],
        )
        try:
            if stage in ("all", "calibrate"):
                prior = calibrate(runner, documents, config, name, model["name"], plan)
            else:
                prior = prepare_prior(runner, documents, config, name, model["name"], plan)
            if stage in ("all", "evaluate"):
                evaluate(runner, documents, config, name, model, plan, prior)
        finally:
            runner.close()
            del runner
        del prior
        gc.collect()
        torch.cuda.empty_cache()
    if stage in ("all", "evaluate"):
        report(name)


def report(name):
    output = RESULTS / (name + "_report.json")
    if not (RESULTS / (name + "_plan.json.gz")).exists() and output.exists():
        value = json.loads(output.read_text(encoding="utf-8"))
        if value.get("historical_subset_export") is True:
            message(f"Historical retained-method report: {output}")
            return value
    plan = read(name + "_plan")
    if not source_matches(name, plan, snapshot()):
        raise ValueError("Restore the frozen source before reporting this run")
    config = plan["config"]
    missing, summaries, selections = [], [], []
    for model in config["models"]:
        for seed in config["seeds"]:
            for method in checkpoint_methods(config):
                identifier = f"{name}_{model['name']}_{method}_{seed}"
                record = existing(identifier, plan)
                if record is None:
                    missing.append(identifier)
                elif method in TRAINED:
                    selections.append(
                        dict(
                            model=model["name"],
                            method=method,
                            seed=seed,
                            selected=record["selected"],
                            selected_last_epoch=record["selected"]["selected_epoch"]
                            == config["epochs"],
                        )
                    )
    family_size = sum(len(required_panels(config, m)) for m in config["models"]) * (
        len(config["methods"]) - 1
    )
    for model in config["models"]:
        for panel in required_panels(config, model):
            records = []
            for seed in config["seeds"]:
                for method in panel["methods"]:
                    identifier = panel_name(name, model["name"], panel, method, seed)
                    record = existing(identifier, plan)
                    if record is None:
                        missing.append(identifier)
                    else:
                        records.extend(
                            dict(**row, method=method, seed=seed) for row in record["observations"]
                        )
            expected = len(panel["methods"]) * len(config["seeds"])
            complete = (
                sum(
                    existing(panel_name(name, model["name"], panel, m, s), plan) is not None
                    for s in config["seeds"]
                    for m in panel["methods"]
                )
                == expected
            )
            if complete and panel["kind"] in ("continuation", "recent", "stream", "task"):
                summary = paired_summary(
                    records,
                    config["seeds"],
                    family_size=family_size,
                    metric="score" if panel["kind"] == "task" else "nll",
                )
                summaries.append(dict(model=model["name"], panel=panel, summary=summary))
                means = {m: round(v, 6) for m, v in summary["means"].items()}
                message(
                    f"{model['name']} {panel['kind']} {panel.get('task', panel['context'])}: {means}"
                )
    # A partial report never impersonates a finished campaign.
    primary = config["primary"]
    primary_assessment = None
    for result in summaries:
        p = result["panel"]
        if (
            result["model"] == primary["model"]
            and p["kind"] == "continuation"
            and p["phase"] == "confirmation"
            and p["context"] == primary["context"]
        ):
            comparison = result["summary"]["comparisons"][primary["comparator"]]
            primary_assessment = dict(
                definition=primary,
                comparison=comparison,
                objective_supported=bool(
                    comparison["gain"] >= primary["minimum_gain"]
                    and comparison["simultaneous_crossed_ci"][0] > 0
                    and all(v > 0 for v in comparison["seed_gains"].values())
                ),
                scope="This criterion does not establish novelty or conference acceptance",
            )
    value = dict(
        plan_sha256=plan["sha256"],
        complete=not missing,
        missing=missing,
        panels=summaries,
        calibration_selections=selections,
        primary_assessment=primary_assessment,
        scope=config["study_scope"],
        publication_ready=False,
    )
    if plan["source"]["sha256"] != snapshot()["sha256"]:
        value["source_repair_sha256"] = read(name + "_source_repair")["sha256"]
    budget_path = RESULTS / (name + "_budget.json")
    value["time_budget"] = json.loads(budget_path.read_text()) if budget_path.exists() else None
    output = RESULTS / (name + "_report.json")
    output.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    message(f"Report: {output}; {len(missing)} missing reader records")
    return value


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        raise SystemExit("Use python -m csd campaign for the supervised study")
    run(sys.argv[1], sys.argv[2], sys.argv[3])
