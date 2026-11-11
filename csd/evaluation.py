"""Actual joint reconstructions, synchronized timings and paired uncertainty."""

import math
import time

import numpy as np
import torch
from scipy.stats import t as student_t

from .runner import compressed_pair, prepared


def resident_bytes(transform):
    """Count unique parameter/buffer/codec allocations, including FP32 rotations."""
    if transform is None:
        return 0
    tensors = []
    if isinstance(transform, torch.nn.Module):
        tensors.extend(transform.parameters())
        tensors.extend(transform.buffers())
    for codec in getattr(transform, "codecs", []):
        tensors.extend([codec.rot, codec.center])
    for name in ("weights", "permutation", "codebooks", "squared"):
        value = getattr(transform, name, None)
        if isinstance(value, torch.Tensor):
            tensors.append(value)
    storages = {
        (str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
        for t in tensors
    }
    return sum(storages.values())


def payload(runner, transform, prefix, batch=1):
    old = max(0, prefix - runner.protected)
    recent = prefix - old
    shape = runner.heads, old, runner.dim
    if transform is None:
        components = dict(key=math.prod(shape) * 2, value=math.prod(shape) * 2)
    elif hasattr(transform, "payload_bytes"):
        components = transform.payload_bytes(shape)
    else:
        components = dict(
            key=(math.prod(shape) + 7) // 8 + runner.heads * old * 2, value=math.prod(shape) * 2
        )
    recent_bytes = recent * runner.heads * runner.dim * 4 * runner.layers
    kv = runner.layers * sum(components.values()) + recent_bytes
    reference = prefix * runner.heads * runner.dim * runner.layers * 4
    decoder = resident_bytes(transform)
    return dict(
        prefix=prefix,
        batch=batch,
        compressed_key_bytes=components["key"] * runner.layers,
        retained_value_bytes=components["value"] * runner.layers,
        recent_bf16_bytes=recent_bytes,
        logical_kv_bytes=kv,
        decoder_resident_bytes=decoder,
        bf16_kv_bytes=reference,
        logical_kv_compression=reference / max(kv, 1),
        net_compression_including_decoder=batch * reference / max(batch * kv + decoder, 1),
        scope="Logical packed payload plus decoder; dense evaluation also materializes BF16 KV",
    )


@torch.no_grad()
def mechanism(runner, transform, article, cut):
    data = runner.data(article, cut)
    keys = [t.cuda() for t in data["keys"]]
    values = [t.cuda() for t in data["values"]]
    queries = runner.trace(article["tokens"][: cut - 1])
    count = keys[0].shape[-2] - runner.protected
    if count <= 0:
        raise ValueError("No compressed keys in mechanism panel")
    errors, energies, score_errors, score_energies, layers = [], [], [], [], []
    with prepared(transform, keys, values, runner.protected):
        for layer, (key, value) in enumerate(zip(keys, values, strict=True)):
            decoded, _ = compressed_pair(key, value, transform, layer, runner.protected)
            source, reconstructed = key[0, :, :count].float(), decoded[0, :, :count].float()
            delta = reconstructed - source
            query = queries[layer][0, :, -64:].cuda().float()
            repeat = query.shape[0] // source.shape[0]
            scores = query @ source.repeat_interleave(repeat, 0).transpose(-1, -2)
            difference = query @ delta.repeat_interleave(repeat, 0).transpose(-1, -2)
            query_positions = torch.arange(cut - 1 - query.shape[-2], cut - 1, device="cuda")
            allowed = torch.arange(count, device="cuda")[None] <= query_positions[:, None]
            e, energy = float(delta.square().sum()), float(source.square().sum())
            se = float(difference.square().masked_select(allowed[None]).sum())
            ss = float(scores.square().masked_select(allowed[None]).sum())
            errors.append(e)
            energies.append(energy)
            score_errors.append(se)
            score_energies.append(ss)
            layers.append(
                dict(
                    layer=layer,
                    key_nmse=e / max(energy, 1e-20),
                    attention_logit_nmse=se / max(ss, 1e-20),
                )
            )
    return dict(
        article=article["sha256"],
        corpus=article["corpus"],
        layers=layers,
        key_nmse=sum(errors) / max(sum(energies), 1e-20),
        attention_logit_nmse=sum(score_errors) / max(sum(score_energies), 1e-20),
    )


@torch.no_grad()
def benchmark(runner, transform, article, cut, warmup=3, repetitions=7):
    data = runner.data(article, cut)
    keys, values = ([t.cuda() for t in data[k]] for k in ("keys", "values"))

    def reconstruction():
        with prepared(transform, keys, values, runner.protected):
            return [
                compressed_pair(k, v, transform, layer, runner.protected)
                for layer, (k, v) in enumerate(zip(keys, values, strict=True))
            ]

    result = {}
    for label, operation in (
        ("encode_reconstruct", reconstruction),
        ("dense_continuation", lambda: runner.forward(data, transform)),
    ):
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        times, events = [], []
        for _ in range(repetitions):
            start, finish = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            torch.cuda.synchronize()
            began = time.perf_counter()
            start.record()
            output = operation()
            finish.record()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - began)
            events.append(start.elapsed_time(finish) / 1000)
            del output
        result[label] = dict(
            wall_seconds=times,
            gpu_seconds=events,
            median_seconds=float(np.median(times)),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            additional_peak_bytes=torch.cuda.max_memory_allocated() - base,
        )
    return dict(
        measurements=result,
        prefix=cut,
        suffix=runner.suffix,
        warmup=warmup,
        repetitions=repetitions,
        article=article["sha256"],
        scope="Dense SDPA reference timings, not compressed-resident serving throughput",
    )


def paired_summary(rows, seeds, metric="nll", family_size=1, draws=100000):
    """Corpus-stratified document clusters; crossed seed/document uncertainty."""
    identifier = "example" if metric == "score" else "article"
    methods = sorted({r["method"] for r in rows})
    ids = sorted({r[identifier] for r in rows})
    table = {(r["method"], r["seed"], r[identifier]): r for r in rows}
    expected = {(m, s, i) for m in methods for s in seeds for i in ids}
    if (
        len(rows) != len(table)
        or set(table) != expected
        or "csd" not in methods
        or not all(np.isfinite(r[metric]) for r in rows)
    ):
        raise ValueError("Incomplete, duplicated or nonfinite paired observations")
    for i in ids:
        inputs = {
            (
                table[m, s, i].get("document", i),
                table[m, s, i].get("corpus", "task"),
                table[m, s, i].get("input_sha256"),
            )
            for m in methods
            for s in seeds
        }
        if len(inputs) != 1:
            raise ValueError("Methods/seeds do not share matched evaluation inputs")
    strata = sorted({r.get("corpus", "task") for r in rows})
    clusters = {}
    for corpus in strata:
        current = [i for i in ids if table[methods[0], seeds[0], i].get("corpus", "task") == corpus]
        documents = sorted({table[methods[0], seeds[0], i].get("document", i) for i in current})
        clusters[corpus] = [
            [i for i in current if table[methods[0], seeds[0], i].get("document", i) == d]
            for d in documents
        ]
    values, counts = {}, {}
    for m in methods:
        values[m] = {}
        for corpus, groups in clusters.items():
            values[m][corpus] = np.array(
                [[sum(table[m, s, i][metric] for i in group) for group in groups] for s in seeds]
            )
            counts[corpus] = np.array([len(group) for group in groups])
    seed_means = {
        m: np.mean([v.sum(-1) / counts[c].sum() for c, v in per.items()], axis=0)
        for m, per in values.items()
    }
    comparisons = {}
    orientation = -1 if metric == "score" else 1
    for comparator in methods:
        if comparator == "csd":
            continue
        differences = {c: orientation * (values[comparator][c] - values["csd"][c]) for c in strata}
        per_seed = orientation * (seed_means[comparator] - seed_means["csd"])
        rng = np.random.default_rng(20261025)
        conditional, crossed = [], []
        for start in range(0, draws, 1000):
            n = min(1000, draws - start)
            selected_seed = rng.integers(len(seeds), size=(n, len(seeds)))
            corpus_conditional, corpus_crossed = [], []
            for c in strata:
                index = rng.integers(len(counts[c]), size=(n, len(counts[c])))
                sampled = differences[c][:, index].sum(-1).T / counts[c][index].sum(-1)[:, None]
                corpus_conditional.append(sampled.mean(-1))
                corpus_crossed.append(np.take_along_axis(sampled, selected_seed, axis=1).mean(-1))
            conditional.extend(np.mean(corpus_conditional, axis=0))
            crossed.extend(np.mean(corpus_crossed, axis=0))
        alpha = 0.05 / max(1, family_size)
        standard_error = per_seed.std(ddof=1) / math.sqrt(len(seeds))
        radius = float(student_t.ppf(0.975, len(seeds) - 1) * standard_error)
        comparisons[comparator] = dict(
            gain=float(per_seed.mean()),
            seed_gains=dict(zip(map(str, seeds), per_seed.tolist())),
            conditional_document_ci95=np.quantile(conditional, [0.025, 0.975]).tolist(),
            crossed_seed_document_ci95=np.quantile(crossed, [0.025, 0.975]).tolist(),
            simultaneous_crossed_ci=np.quantile(crossed, [alpha / 2, 1 - alpha / 2]).tolist(),
            seed_t_ci95=[float(per_seed.mean() - radius), float(per_seed.mean() + radius)],
        )
    return dict(
        means={m: float(v.mean()) for m, v in seed_means.items()},
        seed_means={m: dict(zip(map(str, seeds), v.tolist())) for m, v in seed_means.items()},
        comparisons=comparisons,
        metric=metric,
        bootstrap_draws=draws,
        multiplicity_family=family_size,
        examples=len(ids),
        document_clusters=sum(map(len, clusters.values())),
        uncertainty="Paired corpus-stratified document clusters; three seeds have limited precision",
    )
