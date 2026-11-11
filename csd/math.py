"""Stable Gaussian truncation, half-split RoPE and convex probit inference."""

import math

import torch


def rope(tensor, positions, theta=1_000_000.0, inverse=False):
    dimension = tensor.shape[-1]
    # Match Transformers' reciprocal operation order, including at long positions.
    frequency = 1.0 / (
        theta ** (torch.arange(0, dimension, 2, device=tensor.device).float() / dimension)
    )
    angle = positions.float()[:, None] * frequency[None]
    cosine = torch.cat([angle.cos(), angle.cos()], -1)
    sine = torch.cat([angle.sin(), angle.sin()], -1)
    half = torch.cat([-tensor[..., dimension // 2 :], tensor[..., : dimension // 2]], -1)
    return tensor.float() * cosine + (-1 if inverse else 1) * half * sine


def positive_gaussian_mean(location, variance, signs):
    deviation = variance.clamp_min(1e-12).sqrt()
    standardized = signs * location / deviation
    negative = math.sqrt(2 / math.pi) / torch.special.erfcx(
        (-standardized / math.sqrt(2)).clamp_min(0)
    )
    positive = torch.exp(-standardized.square() / 2) / (
        math.sqrt(2 * math.pi) * torch.special.ndtr(standardized).clamp_min(0.5)
    )
    ratio = torch.where(standardized <= 0, negative, positive)
    magnitude = (signs * location + deviation * ratio).clamp_min(0)
    inverse = (-standardized).clamp_min(8).reciprocal()
    squared = inverse.square()
    tail = (
        deviation
        * inverse
        * (1 + squared * (-2 + squared * (10 + squared * (-74 + squared * 706))))
    )
    return signs * torch.where(standardized < -8, tail, magnitude)


def project_ball(desired, standard):
    center = 0.5 * standard
    radius = 0.5 * standard.norm(dim=-1, keepdim=True)
    direction = desired - center
    distance = direction.norm(dim=-1, keepdim=True)
    return torch.where(
        distance <= radius, desired, center + direction * (radius / distance.clamp_min(1e-20))
    )


def mills_and_curvature(t):
    negative = math.sqrt(2 / math.pi) / torch.special.erfcx((-t / math.sqrt(2)).clamp_min(0))
    positive = torch.exp(-t.square() / 2) / (
        math.sqrt(2 * math.pi) * torch.special.ndtr(t).clamp_min(0.5)
    )
    ratio = torch.where(t <= 0, negative, positive)
    shifted = positive_gaussian_mean(t, torch.ones_like(t), torch.ones_like(t))
    ratio = torch.where(t < -8, -t + shifted, ratio)
    return ratio, (ratio * shifted).clamp(0, 1)


def latent_terms(z, a, b):
    t = a + torch.einsum("...dr,...r->...d", b, z)
    ratio, curvature = mills_and_curvature(t)
    gradient = z - torch.einsum("...dr,...d->...r", b, ratio)
    identity = torch.eye(z.shape[-1], device=z.device, dtype=z.dtype)
    hessian = identity + torch.einsum("...di,...d,...dj->...ij", b, curvature, b)
    return gradient, hessian


def infer_latent(a, b, iterations=10):
    """Probit MAP; stationary implicit derivative, except the one-step ablation."""
    if iterations == 1:
        initial = a.new_zeros(*a.shape[:-1], b.shape[-1])
        gradient, hessian = latent_terms(initial, a, b)
        return initial - torch.linalg.solve(hessian, gradient[..., None])[..., 0]
    with torch.no_grad():
        root = a.new_zeros(*a.shape[:-1], b.shape[-1])
        for step_index in range(40):
            gradient, hessian = latent_terms(root, a, b)
            if step_index >= iterations and float(gradient.abs().max()) < 2e-4:
                break
            step = torch.linalg.solve(hessian, gradient[..., None])[..., 0]
            objective = 0.5 * root.square().sum(-1) - torch.special.log_ndtr(
                a + torch.einsum("...dr,...r->...d", b, root)
            ).sum(-1)
            rate = torch.ones_like(objective)
            # Backtracking per independent token: guard rare large Newton steps.
            for _ in range(16):
                candidate = root - rate[..., None] * step
                value = 0.5 * candidate.square().sum(-1) - torch.special.log_ndtr(
                    a + torch.einsum("...dr,...r->...d", b, candidate)
                ).sum(-1)
                accept = value <= objective + 1e-6 * (1 + objective.abs())
                if bool(accept.all()):
                    break
                rate = torch.where(accept, rate, rate * 0.5)
            root = candidate
        gradient, hessian = latent_terms(root, a, b)
        relative = gradient.abs().amax(-1) / (1 + root.abs().amax(-1))
        if not torch.isfinite(root).all() or float(relative.max()) > 2e-4:
            raise RuntimeError("Probit solver did not converge; implicit gradient is unsafe")
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        gradient, hessian = latent_terms(root, a, b)
        root = (
            root
            - torch.linalg.solve(hessian.detach(), (gradient - gradient.detach())[..., None])[
                ..., 0
            ]
        )
    return root
