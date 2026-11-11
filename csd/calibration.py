"""Shared matched-budget calibration with epoch-zero eligibility and tuning-only selection."""

import json
import time

import numpy as np
import torch


from .progress import message, progress


def fit_decoder(decoder, runner, phases, train_cuts, eval_cuts, seed, lr, epochs, method):
    if epochs < 1 or lr <= 0 or not phases["fit"] or not phases["tune"]:
        raise ValueError("Invalid calibration budget or empty fitting/tuning phase")
    decoder.train()
    best_rows = runner.measure(phases["tune"], eval_cuts, decoder)
    best_score = float(np.mean([r["nll"] for r in best_rows]))
    if not np.isfinite(best_score):
        raise RuntimeError("Non-finite initial tuning loss; no checkpoint accepted")
    initial_score, best_epoch = best_score, 0
    best = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=0)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    events = []
    epochs_bar = progress(range(epochs), desc=f"{method} seed {seed}, LR {lr:g}")
    for epoch in epochs_bar:
        losses, nlls, kls = [], [], []
        for index in progress(
            rng.permutation(len(phases["fit"])),
            desc=f"{method} seed {seed} epoch {epoch + 1}/{epochs}",
            detail=True,
        ):
            for cut in rng.permutation(train_cuts):
                optimizer.zero_grad(set_to_none=True)
                kl, nll = runner.forward(runner.data(phases["fit"][index], int(cut)), decoder)
                loss = nll.mean() + kl
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)
                if not torch.isfinite(norm):
                    raise RuntimeError("Non-finite calibration gradient; no checkpoint accepted")
                optimizer.step()
                if hasattr(decoder, "project_parameters"):
                    decoder.project_parameters()
                losses.append(loss.item())
                nlls.append(nll.mean().item())
                kls.append(kl.item())
        tuning = runner.measure(phases["tune"], eval_cuts, decoder)
        score = float(np.mean([r["nll"] for r in tuning]))
        if not np.isfinite(score):
            raise RuntimeError("Non-finite tuning loss; no checkpoint accepted")
        event = dict(
            method=method,
            seed=seed,
            lr=lr,
            epoch=epoch + 1,
            fit_loss=float(np.mean(losses)),
            fit_nll=float(np.mean(nlls)),
            fit_kl=float(np.mean(kls)),
            tune_nll=score,
            seconds=time.perf_counter() - started,
        )
        events.append(event)
        epochs_bar.set_postfix(fit=f"{event['fit_nll']:.4f}", tune=f"{score:.4f}")
        message(json.dumps(event), detail=True)
        if score < best_score:
            best_score, best_epoch, best_rows = score, epoch + 1, tuning
            best = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
    choice = dict(
        lr=lr,
        tune_nll=best_score,
        initial_tune_nll=initial_score,
        selected_epoch=best_epoch,
        trainable_parameters=sum(p.numel() for p in decoder.parameters() if p.requires_grad),
        tune_rows=best_rows,
        optimizer_updates=epochs * len(phases["fit"]) * len(train_cuts),
    )
    return best, choice, events
