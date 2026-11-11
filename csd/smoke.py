"""Correctness checks only: synthetic tokens never touch reserved evaluation data."""

import gc
import json
import os
from pathlib import Path
import unittest

import numpy as np
import torch


def synthetic_prior(runner, data, group_size=3):
    layers, heads, dim = runner.layers, runner.heads, runner.dim
    width = group_size * heads * dim
    groups = (layers + group_size - 1) // group_size
    return dict(
        centers=torch.stack([k[0].float().mean(-2, keepdim=True) for k in data["keys"]]).cuda(),
        group_predictor=torch.eye(width, device="cuda")[None].repeat(groups, 1, 1) * 0.1,
        group_intercept=torch.zeros(groups, width, device="cuda"),
        factors=torch.randn(layers, heads, 48, dim, device="cuda") * 0.02,
        noise=torch.ones(layers, heads, device="cuda"),
    )


def gpu_check():
    from .campaign import READERS, TRAINED, configure, make
    from .runner import Runner, stream
    from .tasks import greedy

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU smoke requires a CUDA GPU with BF16 support")
    configure()
    config = json.loads(Path("configs/study.json").read_text())
    observations = []
    for index, model in enumerate(config["models"]):
        runner = Runner(
            model["id"], model["revision"], suffix=16, protected=32, reference_cache_gib=1
        )
        rng = np.random.default_rng(239 + index)
        docs = [
            dict(
                sha256=f"synthetic-{index}-{j}",
                corpus="synthetic",
                tokens=rng.integers(1000, 20000, size=300).tolist(),
            )
            for j in range(3)
        ]
        data = runner.data(docs[0], 256)
        torch.manual_seed(11)
        prior = synthetic_prior(runner, data)
        diagonal = torch.full_like(prior["centers"].squeeze(-2), 0.3)
        with torch.no_grad():
            kl, reference = runner.forward(data)
            if float(kl) != 0 or reference.numel() != 16:
                raise AssertionError("BF16/all-target evaluation mismatch")
        methods = (
            READERS
            if index == 0
            else ("bf16", "diagonal", "gaussian", "csd", "headwise", "layerwise")
        )
        for seed in (11, 22, 33):
            for method in methods:
                if seed != 11 and method != "csd":
                    continue
                torch.manual_seed(seed)
                decoder = make(method, prior, diagonal, seed, runner.theta, config)
                with torch.no_grad():
                    kl, nll = runner.forward(data, decoder)
                if nll.numel() != 16 or not torch.isfinite(nll).all() or not torch.isfinite(kl):
                    raise AssertionError(f"Nonfinite {model['name']} {method}")
                if method in TRAINED:
                    optimizer = torch.optim.AdamW(
                        [p for p in decoder.parameters() if p.requires_grad],
                        lr=0.001,
                        weight_decay=0,
                    )
                    optimizer.zero_grad()
                    kl, nll = runner.forward(data, decoder)
                    (nll.mean() + kl).backward()
                    for name, parameter in decoder.named_parameters():
                        if parameter.requires_grad and (
                            parameter.grad is None or not torch.isfinite(parameter.grad).all()
                        ):
                            raise AssertionError(f"Missing/nonfinite {method} {name} gradient")
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)
                    optimizer.step()
                    if hasattr(decoder, "project_parameters"):
                        decoder.project_parameters()
                    del optimizer
                if method in ("csd", "gaussian") and seed == 11:
                    losses, _, teacher = stream(runner.model, docs[0]["tokens"], 192, 3, None, 32)
                    a, b, _ = stream(runner.model, docs[0]["tokens"], 192, 3, decoder, 32, teacher)
                    if not np.isfinite([*losses, *a, *b]).all():
                        raise AssertionError("Nonfinite joint sequential evaluation")
                    answer = greedy(runner.model, docs[0]["tokens"][:192], decoder, 2, set(), 32)
                    if len(answer) != 2:
                        raise AssertionError(
                            "Generation smoke did not cover first and second tokens"
                        )
                observations.append(
                    dict(
                        model=model["name"],
                        method=method,
                        seed=seed,
                        scored_targets=16,
                        finite=True,
                    )
                )
                del decoder
        del prior, data
        runner.close()
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    print(
        json.dumps(
            dict(
                gpu_correctness="passed",
                reader_checks=len(observations),
                backbones=sorted({r["model"] for r in observations}),
                final_test_scored=False,
            ),
            indent=2,
        )
    )


def run(gpu=False):
    os.environ.setdefault("CSD_PROGRESS", "0")
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    if gpu:
        gpu_check()
