"""Published scalar baselines with explicit implementation provenance."""

import math

import torch

from .quantization import RotatedCodec


class Eden:
    def __init__(self, centers, seed, mode):
        self.mode = mode
        self.codecs = [
            RotatedCodec(
                dimension=centers.shape[-1],
                heads=centers.shape[1],
                seed=seed + 1000003 * layer,
                mode="unbiased" if mode == "unbiased" else "optimal",
                mixed=False,
                bits=1,
                device=centers.device,
                center=center,
            )
            for layer, center in enumerate(centers)
        ]

    def __call__(self, key, layer):
        codec = self.codecs[layer]
        cache = codec.encode(key)
        if self.mode == "matched_norm":
            # At one bit every centroid has the same magnitude.
            cache.scale = (
                (key.float() - codec.center).norm(dim=-1, keepdim=True) / math.sqrt(codec.dimension)
            ).half()
        return codec.decode(cache).to(key.dtype)


class Hqq:
    """Official HQQ scalar quantizer, adapted to token-wise key-only compression.

    Groups contain 64 feature coordinates; actual scale/zero dtypes are charged.
    Cached prefixes are immutable within a sequence; verify before reuse.
    This is a reference KV adaptation, not a claim about HQQ serving kernels.
    """

    def __init__(self):
        from hqq.core.quantize import Quantizer

        self.quantizer = Quantizer
        self.metadata_bytes = 4
        self.start_sequence()

    def start_sequence(self):
        self.prefixes = {}

    def quantize(self, key):
        if key.shape[-1] % 64:
            raise ValueError("HQQ control requires feature groups of 64")
        packed, metadata = self.quantizer.quantize(
            key.float(),
            nbits=1,
            group_size=64,
            axis=1,
            optimize=True,
            # Upstream packs one-bit groups along axis zero in blocks of eight.
            # A small append can have fewer groups. Unpacked reference codes
            # give identical reconstructions without padding artificial data.
            bitpack=(key.numel() // 64) % 8 == 0,
            view_as_float=False,
            device=str(key.device),
        )
        self.metadata_bytes = metadata["scale"].element_size() + metadata["zero"].element_size()
        return self.quantizer.dequantize(packed, metadata).to(key.dtype)

    def pair(self, key, value, layer=0):
        previous = self.prefixes.get(layer)
        count = 0
        if previous is not None:
            raw, decoded = previous
            if (
                raw.shape[0] == key.shape[0]
                and raw.shape[-1] == key.shape[-1]
                and raw.shape[-2] <= key.shape[-2]
                and torch.equal(raw, key[:, : raw.shape[-2]])
            ):
                count = raw.shape[-2]
        if count == key.shape[-2] and previous is not None:
            return decoded, value
        tail = self.quantize(key[:, count:])
        output = torch.cat([decoded, tail], -2) if count else tail
        self.prefixes[layer] = key.detach().clone(), output.detach()
        return output, value

    def payload_bytes(self, shape):
        count = math.prod(shape)
        return dict(key=math.ceil(count / 8) + count // 64 * self.metadata_bytes, value=count * 2)


def provenance():
    import importlib.metadata
    import inspect
    from hqq.core.quantize import Quantizer
    from .artifacts import file_hash

    return dict(
        hqq=dict(
            repository="https://github.com/dropbox/hqq",
            version=importlib.metadata.version("hqq"),
            source_sha256=file_hash(inspect.getfile(Quantizer)),
            adapter="Official optimized quantizer; our key-only adaptation",
        ),
        eden_drive=dict(execution="Independent one-bit formula implementations"),
        scope="Dense quality comparisons, not authors' packed serving engines",
        not_implemented=["Complete AQUA-KV", "Complete QJL serving system", "RISK-KV"],
    )
