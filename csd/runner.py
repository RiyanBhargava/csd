"""Matched dense-cache evaluation; all suffix targets, bounded teacher caching."""

import copy
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import torch

from .progress import progress


def compressed_pair(key, value, transform, layer, protected=128):
    count = key.shape[-2] - protected
    if transform is None or count <= 0:
        return key, value
    old_key, old_value = key[0, :, :count], value[0, :, :count]
    if hasattr(transform, "pair"):
        new_key, new_value = transform.pair(old_key, old_value, layer)
    elif hasattr(transform, "with_values"):
        positions = torch.arange(count, device=key.device)
        new_key, new_value = transform.with_values(old_key, old_value, layer, positions), old_value
    else:
        new_key, new_value = transform(old_key, layer), old_value
    return (
        torch.cat([new_key[None].to(key.dtype), key[..., count:, :]], -2),
        torch.cat([new_value[None].to(value.dtype), value[..., count:, :]], -2),
    )


@contextmanager
def prepared(transform, keys, values, protected):
    if hasattr(transform, "bind_cache"):
        transform.bind_cache(keys, values, protected)
    try:
        if hasattr(transform, "start_sequence"):
            transform.start_sequence()
        yield
    finally:
        if hasattr(transform, "release_cache"):
            transform.release_cache()


@torch.no_grad()
def prefill(model, tokens, chunk=256):
    if not tokens:
        raise ValueError("Prefill requires at least one token")
    cache = None
    for first in range(0, len(tokens), chunk):
        output = model(
            torch.tensor([tokens[first : first + chunk]], device=model.device),
            past_key_values=cache,
            use_cache=True,
        )
        cache = output.past_key_values
    return cache


class Runner:
    def __init__(
        self, model, revision, suffix=64, cache_windows=128, protected=128, reference_cache_gib=12
    ):
        from transformers import AutoModelForCausalLM
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        if suffix < 1 or protected < 1:
            raise ValueError("Positive suffix and recent-window budgets required")
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model, revision=revision, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        config = self.model.config
        if (
            config.model_type not in ("qwen2", "qwen3", "llama")
            or getattr(config, "rope_scaling", None)
            or getattr(config, "use_sliding_window", False)
        ):
            raise ValueError("Only unscaled half-split RoPE Qwen2/Qwen3/Llama are supported")
        self.layers, self.heads = config.num_hidden_layers, config.num_key_value_heads
        self.dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.theta, self.suffix, self.protected = config.rope_theta, suffix, protected
        self.identity = dict(
            model=model,
            revision=revision,
            suffix=suffix,
            dtype="bfloat16",
            attention="sdpa_math",
            schema=3,
            score="All suffix targets; prefill prefix minus final token",
        )
        self._references = OrderedDict()
        self._reference_bytes, self._reference_limit = 0, int(reference_cache_gib * 1024**3)
        self._reference_windows = cache_windows
        self._capture = None

        def attention(module, query, key, value, attention_mask, **kwargs):
            if self._capture is not None:
                self._capture.setdefault(module.layer_idx, []).append(query.detach().cpu())
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

        self._attention = attention
        ALL_ATTENTION_FUNCTIONS["csd_math"] = attention
        config._attn_implementation = "csd_math"

    def _data(self, article, cut):
        tokens = article["tokens"][: cut + self.suffix]
        if cut < 2 or len(tokens) != cut + self.suffix:
            raise ValueError("Continuation exceeds the document")
        with torch.no_grad():
            original = prefill(self.model, tokens[: cut - 1])
            teacher = self.model(
                torch.tensor([tokens[cut - 1 : -1]], device="cuda"),
                past_key_values=copy.deepcopy(original),
                use_cache=True,
            )
            return dict(
                keys=[t.detach().cpu() for t in original.key_cache],
                values=[t.detach().cpu() for t in original.value_cache],
                logp=teacher.logits[0].float().log_softmax(-1).cpu(),
                tokens=tokens,
                prefix=cut,
                identity=self.identity,
            )

    def data(self, article, cut):
        key = article["sha256"], cut
        if key in self._references:
            self._references.move_to_end(key)
            return self._references[key][0]
        data = self._data(article, cut)
        size = (
            sum(t.numel() * t.element_size() for name in ("keys", "values") for t in data[name])
            + data["logp"].numel() * data["logp"].element_size()
        )
        # CPU-only bounded LRU: no unbounded multi-model teacher files on disk.
        if self._reference_windows > 0 and size <= self._reference_limit:
            while self._references and (
                len(self._references) >= self._reference_windows
                or self._reference_bytes + size > self._reference_limit
            ):
                _, (_, removed) = self._references.popitem(last=False)
                self._reference_bytes -= removed
            self._references[key] = data, size
            self._reference_bytes += size
        return data

    def forward(self, data, transform=None):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        with prepared(transform, data["keys"], data["values"], self.protected):
            for layer, (key, value) in enumerate(zip(data["keys"], data["values"], strict=True)):
                key, value = compressed_pair(
                    key.to(self.model.device),
                    value.to(self.model.device),
                    transform,
                    layer,
                    self.protected,
                )
                cache.update(key, value, layer)
            cut = data["prefix"]
            tokens = torch.tensor([data["tokens"]], device=self.model.device)
            output = self.model(tokens[:, cut - 1 : -1], past_key_values=cache, use_cache=True)
            logp = output.logits[0].float().log_softmax(-1)
            reference = data["logp"].to(self.model.device)
            kl = (reference.exp() * (reference - logp)).sum(-1).mean()
            nll = -logp.gather(-1, tokens[0, cut:, None]).squeeze(-1)
            return kl, nll

    @torch.no_grad()
    def measure(self, articles, cuts, transform=None):
        rows = []
        for article in progress(articles, desc="Articles", detail=True):
            windows = []
            for cut in cuts:
                kl, nll = self.forward(self.data(article, cut), transform)
                if nll.numel() != self.suffix or not torch.isfinite(nll).all():
                    raise RuntimeError("Invalid scored target coverage")
                windows.append(
                    dict(prefix=cut, nll=float(nll.mean()), kl=float(kl), token_nll=nll.tolist())
                )
            rows.append(
                dict(
                    article=article["sha256"],
                    corpus=article["corpus"],
                    nll=float(np.mean([w["nll"] for w in windows])),
                    kl=float(np.mean([w["kl"] for w in windows])),
                    windows=windows,
                )
            )
        return rows

    @torch.no_grad()
    def trace(self, tokens):
        """Native post-RoPE prefix queries for causal attention-logit diagnostics."""
        self._capture = {}
        try:
            prefill(self.model, tokens)
            return [torch.cat(self._capture[layer], -2) for layer in range(self.layers)]
        finally:
            self._capture = None

    def close(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self._references.clear()
        self._reference_bytes = 0
        if ALL_ATTENTION_FUNCTIONS.get("csd_math") is self._attention:
            ALL_ATTENTION_FUNCTIONS.pop("csd_math")


@contextmanager
def intervention(model, transform, protected=128):
    """Prepare all *past* layers before a one-token forward advances any layer."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    previous = model.config._attn_implementation
    state = dict(enabled=False)
    if hasattr(transform, "start_sequence") and not hasattr(transform, "bind_cache"):
        transform.start_sequence()

    def before_forward(module, args, kwargs):
        if not state["enabled"]:
            return
        cache = kwargs["past_key_values"]
        if hasattr(transform, "bind_cache"):
            # Current token is one of the protected tokens, but isn't in cache yet.
            transform.bind_cache(tuple(cache.key_cache), tuple(cache.value_cache), protected - 1)
            transform.start_sequence()

    def attention(module, query, key, value, attention_mask, **kwargs):
        if state["enabled"]:
            key, value = compressed_pair(key, value, transform, module.layer_idx, protected)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

    handle = model.register_forward_pre_hook(before_forward, with_kwargs=True)
    ALL_ATTENTION_FUNCTIONS["csd_stream"] = attention
    model.config._attn_implementation = "csd_stream"
    try:
        yield state
    finally:
        handle.remove()
        if hasattr(transform, "release_cache"):
            transform.release_cache()
        model.config._attn_implementation = previous
        ALL_ATTENTION_FUNCTIONS.pop("csd_stream", None)


@torch.no_grad()
def stream(model, tokens, prefix, steps, transform, protected=128, teacher=None):
    if prefix < 2 or prefix + steps > len(tokens):
        raise ValueError("Invalid sequential continuation window")
    losses, divergences, references = [], [], []
    with intervention(model, transform, protected) as state:
        cache = prefill(model, tokens[: prefix - 1])
        state["enabled"] = True
        for index in range(steps):
            output = model(
                torch.tensor([[tokens[prefix - 1 + index]]], device=model.device),
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            logp = output.logits[0, -1].float().log_softmax(-1)
            losses.append(float(-logp[tokens[prefix + index]]))
            if teacher is None:
                references.append(logp.cpu())
            else:
                reference = teacher[index].to(logp.device)
                divergences.append(float((reference.exp() * (reference - logp)).sum()))
    return losses, divergences, references
