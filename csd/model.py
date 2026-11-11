"""One canonical cooperative decoder, component ablations and internal controls."""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .math import infer_latent, positive_gaussian_mean, project_ball, rope
from .quantization import RotatedCodec

ABLATIONS = ("headwise", "layerwise", "no_correlation", "no_position", "one_step")
TRAINED = (
    "diagonal",
    "continued_diagonal",
    "signed",
    "linear",
    "neural",
    "csd",
    "gaussian",
    "value_only",
    *ABLATIONS,
)


class DiagonalDecoder(nn.Module):
    def __init__(self, centers, seed=11, theta=1_000_000.0):
        super().__init__()
        self.register_buffer("centers", centers.detach(), persistent=False)
        self.diagonal = nn.Parameter(torch.zeros_like(centers.squeeze(-2)))
        self.dim, self.seed, self.theta = centers.shape[-1], seed, theta
        self.codecs = [
            RotatedCodec(
                self.dim,
                centers.shape[1],
                seed + 1000003 * layer,
                device=centers.device,
                center=center,
            )
            for layer, center in enumerate(centers)
        ]

    def forward(self, key, layer):
        codec = self.codecs[layer]
        code = codec.vector(codec.encode(key.detach()))
        decoded = code * (1 + self.diagonal[layer].clamp_min(0)).reciprocal()[:, None]
        return (decoded @ codec.rot.transpose(-1, -2) + codec.center).to(key.dtype)

    @torch.no_grad()
    def project_parameters(self):
        self.diagonal.clamp_(min=0)

    def payload_bytes(self, shape):
        heads, tokens, dim = shape
        return dict(
            key=math.ceil(heads * tokens * dim / 8) + heads * tokens * 2,
            value=heads * tokens * dim * 2,
        )


class CorrectionDecoder(DiagonalDecoder):
    """Strong signed-coordinate and full-linear task calibration controls."""

    def __init__(self, centers, diagonal, seed, theta, kind):
        super().__init__(centers, seed, theta)
        self.diagonal.data.copy_(diagonal)
        self.diagonal.requires_grad_(False)
        self.kind = kind
        if kind == "linear":
            self.correction = nn.Parameter(
                torch.zeros(*diagonal.shape, self.dim, device=diagonal.device)
            )
        elif kind == "signed":
            gain = (1 + diagonal).reciprocal()
            self.positive, self.negative = nn.Parameter(gain.clone()), nn.Parameter(gain.clone())
            del self.diagonal
        else:
            raise ValueError("Unknown correction control")

    def forward(self, key, layer):
        codec = self.codecs[layer]
        standard = codec.vector(codec.encode(key.detach()))
        if self.kind == "signed":
            gain = torch.where(
                standard >= 0, self.positive[layer, :, None], self.negative[layer, :, None]
            )
            result = project_ball(standard * gain, standard)
        else:
            anchor = standard * (1 + self.diagonal[layer]).reciprocal()[:, None]
            result = anchor + anchor @ self.correction[layer]
        return (result @ codec.rot.transpose(-1, -2) + codec.center).to(key.dtype)

    @torch.no_grad()
    def project_parameters(self):
        if self.kind == "signed":
            self.positive.clamp_(0, 2)
            self.negative.clamp_(0, 2)


class NeuralDecoder(DiagonalDecoder):
    """Internal nonlinear sign-pattern decoder, with more capacity than CSD."""

    def __init__(self, centers, diagonal, seed, theta, hidden=16):
        super().__init__(centers, seed, theta)
        self.diagonal.data.copy_(diagonal)
        self.diagonal.requires_grad_(False)
        layers, heads, dim = diagonal.shape
        self.first = nn.Parameter(
            torch.randn(layers, heads, 2 * dim, hidden, device=centers.device) / math.sqrt(2 * dim)
        )
        self.second = nn.Parameter(torch.zeros(layers, heads, hidden, dim, device=centers.device))

    def forward(self, key, layer):
        codec = self.codecs[layer]
        cache = codec.encode(key.detach())
        standard = codec.vector(cache)
        sign = torch.where(standard >= 0, 1.0, -1.0)
        features = torch.cat([sign, sign.abs()], -1)
        correction = torch.nn.functional.gelu(features @ self.first[layer]) @ self.second[layer]
        anchor = standard * (1 + self.diagonal[layer]).reciprocal()[:, None]
        result = project_ball(anchor + 0.5 * correction.tanh() * cache.scale.float(), standard)
        return (result @ codec.rot.transpose(-1, -2) + codec.center).to(key.dtype)


def group_flatten(tensor, group_size, rank=False):
    """Pad incomplete groups with zero evidence, never with fabricated signs."""
    layers, heads, tokens, dim = tensor.shape[:4]
    groups = (layers + group_size - 1) // group_size
    if layers < groups * group_size:
        tensor = torch.cat(
            [tensor, tensor.new_zeros(groups * group_size - layers, *tensor.shape[1:])]
        )
    if rank:
        return (
            tensor.reshape(groups, group_size, heads, tokens, dim, tensor.shape[-1])
            .permute(0, 3, 1, 2, 4, 5)
            .reshape(groups, tokens, -1, tensor.shape[-1])
        )
    return (
        tensor.reshape(groups, group_size, heads, tokens, dim)
        .permute(0, 3, 1, 2, 4)
        .reshape(groups, tokens, -1)
    )


def group_expand(latent, layers, heads, group_size):
    return (
        latent[:, None, None]
        .expand(-1, group_size, heads, -1, -1)
        .reshape(-1, heads, latent.shape[-2], latent.shape[-1])[:layers]
    )


def gaussian_estimate(mean, noise, basis, observed, quantization_variance, group_size=None):
    """Woodbury Gaussian conditioning, joint over the leading layer/head axes."""
    layers, heads, tokens, dim = mean.shape
    rank = basis.shape[-1]
    group_size = layers if group_size is None else group_size
    total = noise + quantization_variance
    a = group_flatten((observed - mean) / total.sqrt(), group_size)
    b = group_flatten(basis / total.sqrt()[..., None], group_size, rank=True)
    hessian = torch.eye(rank, device=b.device, dtype=b.dtype) + b.transpose(-1, -2) @ b
    latent = torch.linalg.solve(hessian, b.transpose(-1, -2) @ a[..., None])[..., 0]
    latent = group_expand(latent, layers, heads, group_size)
    conditional = mean + torch.einsum("lhndr,lhnr->lhnd", basis, latent)
    return conditional + noise / total * (observed - conditional)


class CooperativeDecoder(DiagonalDecoder):
    """Decode complete already-past groups; no hidden per-layer fallback exists."""

    def __init__(
        self,
        prior,
        diagonal,
        seed=11,
        theta=1_000_000.0,
        kind="csd",
        group_size=3,
        token_chunk=64,
        checkpoint_chunks=True,
    ):
        if kind not in ("csd", "gaussian", "value_only", *ABLATIONS):
            raise ValueError("Unknown cooperative decoder component setting")
        super().__init__(prior["centers"], seed, theta)
        self.diagonal.data.copy_(diagonal)
        self.diagonal.requires_grad_(False)
        self.kind, self.group_size, self.token_chunk = kind, group_size, token_chunk
        self.checkpoint_chunks = checkpoint_chunks
        names = (
            ("group_predictor", "group_intercept")
            if kind == "value_only"
            else ("group_predictor", "group_intercept", "factors", "noise")
        )
        for name in names:
            self.register_buffer(name, prior[name].detach(), persistent=False)
        self._bound, self.prepared = None, {}
        if kind == "value_only":
            self.log_gain = nn.Parameter(torch.zeros_like(diagonal))
            self.bias = nn.Parameter(torch.zeros_like(diagonal))
            del self.diagonal
        else:
            self.blend = nn.Parameter(torch.full_like(diagonal, 0.5))
            self.log_variance = nn.Parameter(torch.zeros_like(diagonal))
            self.coupling_logit = nn.Parameter(
                torch.zeros_like(self.noise), requires_grad=kind != "no_correlation"
            )
            if kind == "gaussian":
                # Extra covariance and observation-noise freedom favors the control.
                self.quantization_log_variance = nn.Parameter(torch.zeros_like(diagonal))

    def forward(self, key, layer):
        raise ValueError("CSD needs the complete already-past cache; bind_cache first")

    def bind_cache(self, keys, values, protected):
        if (
            len(keys) != len(self.codecs)
            or len(values) != len(keys)
            or protected < 0
            or len({t.shape[-2] for t in [*keys, *values]}) != 1
        ):
            raise ValueError("Complete aligned layer caches and nonnegative protection required")
        if any(k.shape != v.shape or k.shape[0] != 1 for k, v in zip(keys, values, strict=True)):
            raise ValueError("Single-sequence aligned GQA cache required")
        # Snapshot lists: DynamicCache replaces entries as individual layers advance.
        self._bound = tuple(keys), tuple(values), protected

    def release_cache(self):
        self._bound = None
        self.prepared.clear()

    def conditional_prior(self, values, positions):
        heads, tokens, dim = values[0].shape
        means = []
        for group, first in enumerate(range(0, len(values), self.group_size)):
            current = values[first : first + self.group_size]
            width = len(current) * heads * dim
            x = torch.cat(
                [v.detach().float().transpose(0, 1).reshape(tokens, -1) for v in current], -1
            )
            prediction = (
                x @ self.group_predictor[group, :width, :width]
                + self.group_intercept[group, :width]
            )
            means.extend(
                prediction.reshape(tokens, len(current), heads, dim).permute(1, 2, 0, 3).unbind()
            )
        return torch.stack(
            [
                (rope(mean, positions, self.theta) - codec.center) @ codec.rot
                for mean, codec in zip(means, self.codecs, strict=True)
            ]
        )

    def basis(self, positions, layer):
        basis = self.factors[layer, :, :, None, :].expand(-1, -1, len(positions), -1)
        if self.kind != "no_position":
            basis = rope(basis, positions, self.theta)
        return (basis @ self.codecs[layer].rot[:, None]).permute(0, 2, 3, 1)

    def _chunk(self, mean, standard, positions):
        signs = torch.where(standard >= 0, 1.0, -1.0)
        raw = torch.stack([self.basis(positions, layer) for layer in range(len(self.codecs))])
        calibration = self.log_variance[:, :, None].exp()
        coupling = self.coupling_logit[:, :, None, None].sigmoid()
        basis = raw * coupling.sqrt()[..., None] * calibration.sqrt()[..., None]
        variance = (
            self.noise[:, :, None, None] + (1 - coupling) * raw.square().sum(-1)
        ) * calibration
        if self.kind == "gaussian":
            observation_noise = (
                standard.square()
                * (1 - 2 / torch.pi)
                * self.quantization_log_variance[:, :, None].exp()
            ).clamp_min(1e-12)
            estimate = gaussian_estimate(
                mean, variance, basis, standard, observation_noise, self.group_size
            )
            estimate = signs * (signs * estimate).clamp_min(0)
        elif self.kind == "no_correlation":
            estimate = positive_gaussian_mean(mean, variance + basis.square().sum(-1), signs)
        else:
            a, b = (
                signs * mean / variance.sqrt(),
                signs[..., None] * basis / variance.sqrt()[..., None],
            )
            observed = standard.abs().sum(-1, keepdim=True) > 0
            a, b = torch.where(observed, a, 0), torch.where(observed[..., None], b, 0)
            layers, heads, tokens, dim = a.shape
            rank = b.shape[-1]
            if self.kind == "headwise":
                latent = infer_latent(a, b)
            elif self.kind == "layerwise":
                latent = infer_latent(
                    a.permute(0, 2, 1, 3).reshape(layers, tokens, -1),
                    b.permute(0, 2, 1, 3, 4).reshape(layers, tokens, -1, rank),
                )
                latent = latent[:, None].expand(-1, heads, -1, -1)
            else:
                latent = infer_latent(
                    group_flatten(a, self.group_size),
                    group_flatten(b, self.group_size, rank=True),
                    iterations=1 if self.kind == "one_step" else 10,
                )
                latent = group_expand(latent, layers, heads, self.group_size)
            conditional = mean + torch.einsum("lhndr,lhnr->lhnd", basis, latent)
            estimate = positive_gaussian_mean(conditional, variance, signs)
        anchor = standard * (1 + self.diagonal).reciprocal()[:, :, None]
        return project_ball(anchor + self.blend[:, :, None] * (estimate - anchor), standard)

    def decode_all(self, caches, values, positions):
        mean = self.conditional_prior(values, positions)
        if self.kind == "value_only":
            result = mean * self.log_gain[:, :, None].exp() + self.bias[:, :, None]
        else:
            standard = torch.stack(
                [codec.vector(cache) for codec, cache in zip(self.codecs, caches, strict=True)]
            )
            pieces = []
            for first in range(0, len(positions), self.token_chunk):
                sl = slice(first, first + self.token_chunk)
                inputs = mean[:, :, sl], standard[:, :, sl], positions[sl]
                if self.checkpoint_chunks and torch.is_grad_enabled():
                    pieces.append(
                        checkpoint(
                            self._chunk, *inputs, use_reentrant=False, preserve_rng_state=False
                        )
                    )
                else:
                    pieces.append(self._chunk(*inputs))
            result = torch.cat(pieces, -2)
        return torch.stack(
            [
                result[layer] @ codec.rot.transpose(-1, -2) + codec.center
                for layer, codec in enumerate(self.codecs)
            ]
        )

    def start_sequence(self):
        self.prepared.clear()
        if self._bound is None:
            raise ValueError("Bind the complete already-past cache first")
        keys, values, protected = self._bound
        count = keys[0].shape[-2] - protected
        if count <= 0:
            return
        device = self.centers.device
        retained = [v[0, :, :count].detach().to(device) for v in values]
        # The zero-key control must not even encode the keys.
        caches = (
            None
            if self.kind == "value_only"
            else [
                codec.encode(key[0, :, :count].detach().to(device))
                for codec, key in zip(self.codecs, keys, strict=True)
            ]
        )
        decoded = self.decode_all(caches, retained, torch.arange(count, device=device))
        self.prepared = {layer: value.to(keys[layer].dtype) for layer, value in enumerate(decoded)}

    def with_values(self, key, values, layer, positions=None):
        if layer not in self.prepared or self.prepared[layer].shape != key.shape:
            raise ValueError("Prepared group reconstruction does not match the old prefix")
        return self.prepared[layer]

    def payload_bytes(self, shape):
        if self.kind == "value_only":
            return dict(key=0, value=math.prod(shape) * 2)
        return super().payload_bytes(shape)

    @torch.no_grad()
    def project_parameters(self):
        if self.kind == "value_only":
            self.log_gain.clamp_(-4, 4)
            self.bias.clamp_(-4, 4)
        else:
            self.blend.clamp_(0, 1)
            self.log_variance.clamp_(-4, 4)
            self.coupling_logit.clamp_(-8, 8)
            if self.kind == "gaussian":
                self.quantization_log_variance.clamp_(-4, 4)
