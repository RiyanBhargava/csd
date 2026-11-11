"""Packed one-bit EDEN/DRIVE formula implementations (not serving kernels)."""

from dataclasses import dataclass
import math

import torch


def pack_codes(codes, bits=1):
    if bits != 1 or torch.is_floating_point(codes) or codes.dtype == torch.bool:
        raise ValueError("Integer one-bit codes required")
    if codes.numel() and ((codes < 0).any() or (codes > 1).any()):
        raise ValueError("Codes must be zero or one")
    flat = codes.flatten().long()
    positions = torch.arange(flat.numel(), device=codes.device)
    packed = torch.zeros((flat.numel() + 7) // 8, dtype=torch.long, device=codes.device)
    packed.scatter_add_(0, positions // 8, flat << (positions % 8))
    return packed.byte()


def unpack_codes(packed, bits, shape):
    count = math.prod(shape)
    if bits != 1 or packed.dtype != torch.uint8 or packed.ndim != 1:
        raise ValueError("Packed one-bit buffer required")
    if packed.numel() != (count + 7) // 8:
        raise ValueError("Packed length does not match shape")
    positions = torch.arange(count, device=packed.device)
    return ((packed[positions // 8].long() >> (positions % 8)) & 1).reshape(shape)


def rotation(dimension, seed, device):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(matrix)
    return (q * r.diagonal().sign()).float().to(device)


@dataclass
class RotatedCache:
    codes: list
    scale: torch.Tensor
    shape: tuple

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in [*self.codes, self.scale])


class RotatedCodec:
    def __init__(
        self,
        dimension,
        heads,
        seed,
        device="cpu",
        center=None,
        mode="unbiased",
        mixed=False,
        bits=1,
    ):
        if bits != 1 or mixed or mode not in ("unbiased", "optimal"):
            raise ValueError("This codec implements only one-bit EDEN/DRIVE")
        self.dimension, self.mode = dimension, mode
        self.rot = torch.stack([rotation(dimension, seed + h * 1009, device) for h in range(heads)])
        self.center = (
            torch.zeros(heads, 1, dimension, device=device) if center is None else center.detach()
        )
        if self.center.shape != (heads, 1, dimension):
            raise ValueError("Center layout mismatch")

    def vector(self, cache):
        signs = unpack_codes(cache.codes[0], 1, cache.shape).float() * 2 - 1
        return signs * cache.scale.float()

    def encode(self, key):
        if (
            key.ndim != 3
            or key.shape[0] != len(self.rot)
            or key.shape[-1] != self.dimension
            or not torch.isfinite(key).all()
        ):
            raise ValueError("Invalid key tensor")
        rotated = (key.float() - self.center) @ self.rot
        scale = (
            rotated.square().sum(-1, keepdim=True)
            / rotated.abs().sum(-1, keepdim=True).clamp_min(1e-20)
            if self.mode == "unbiased"
            else rotated.abs().mean(-1, keepdim=True)
        )
        if not torch.isfinite(scale.half()).all():
            raise ValueError("FP16 scale overflow")
        return RotatedCache([pack_codes((rotated >= 0).long())], scale.half(), tuple(key.shape))

    def decode(self, cache):
        return self.vector(cache) @ self.rot.transpose(-1, -2) + self.center

    def __call__(self, key):
        return self.decode(self.encode(key)).to(key.dtype)
