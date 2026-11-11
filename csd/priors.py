"""Model-independent fitting of conditional residual factors in depth groups."""

import torch
from torch.nn import functional as F

from .math import rope

from .progress import progress


def ridge_from_moments(mean_v, mean_k, vv, vk, strength=0.01):
    covariance = vv - mean_v[:, None] * mean_v[None]
    cross = vk - mean_v[:, None] * mean_k[None]
    ridge = strength * covariance.diagonal().mean().clamp_min(1e-8)
    identity = torch.eye(covariance.shape[-1], device=covariance.device, dtype=covariance.dtype)
    matrix = torch.linalg.solve(covariance + ridge * identity, cross)
    return matrix, mean_k - mean_v @ matrix


@torch.no_grad()
def fit_structured_prior(runner, articles, cut, group_size=3, rank_per_layer=16):
    if len(articles) < 2 or group_size < 1 or rank_per_layer < 1:
        raise ValueError("Cross-fitting requires at least two articles and a valid layout")
    max_width = group_size * runner.heads * runner.dim
    max_rank = group_size * rank_per_layer
    predictors, intercepts, factors, noises, diagnostics = [], [], [], [], []
    for first in progress(
        range(0, runner.layers, group_size), desc="Fit conditional depth groups", leave=True
    ):
        count_layers = min(group_size, runner.layers - first)
        width = count_layers * runner.heads * runner.dim
        rank = min(count_layers * rank_per_layer, width - 1)
        xx, yy = [], []
        for article in articles:
            data = runner.data(article, cut)
            keys = [data["keys"][layer][0].float() for layer in range(first, first + count_layers)]
            tokens = keys[0].shape[-2]
            xx.append(
                torch.cat(
                    [
                        data["values"][layer][0].float().transpose(0, 1).reshape(tokens, -1)
                        for layer in range(first, first + count_layers)
                    ],
                    -1,
                )
            )
            yy.append(
                torch.cat(
                    [
                        rope(k, torch.arange(tokens), runner.theta, inverse=True)
                        .transpose(0, 1)
                        .reshape(tokens, -1)
                        for k in keys
                    ],
                    -1,
                )
            )
        moments = [
            (x.sum(0), y.sum(0), x.T @ x, x.T @ y, len(x)) for x, y in zip(xx, yy, strict=True)
        ]
        total = [sum(block[j] for block in moments) for j in range(5)]
        matrix, offset = ridge_from_moments(*[total[j] / total[4] for j in range(4)])
        predictors.append(F.pad(matrix, (0, max_width - width, 0, max_width - width)))
        intercepts.append(F.pad(offset, (0, max_width - width)))
        residuals = []
        for (x, y), block in zip(zip(xx, yy, strict=True), moments, strict=True):
            n = total[4] - block[4]
            loo_matrix, loo_offset = ridge_from_moments(
                *[(total[j] - block[j]) / n for j in range(4)]
            )
            residuals.append(
                (y - x @ loo_matrix - loo_offset).reshape(
                    -1, count_layers, runner.heads, runner.dim
                )
            )
        residual = torch.cat(residuals)
        normalization = residual.square().mean((0, 3), keepdim=True).clamp_min(1e-8).sqrt()
        standardized = (residual / normalization).reshape(len(residual), -1)
        covariance = standardized.double().T @ standardized.double() / len(standardized)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        noise = eigenvalues[:-rank].mean().clamp_min(1e-8)
        modes = eigenvectors[:, -rank:] * (eigenvalues[-rank:] - noise).clamp_min(0).sqrt()[None]
        modes = (
            modes.reshape(count_layers, runner.heads, runner.dim, rank).permute(0, 1, 3, 2).float()
        )
        modes = modes * normalization[0, :, :, :, None]
        factors.append(F.pad(modes, (0, 0, 0, max_rank - rank)))
        noises.append(noise.float() * normalization[0, :, :, 0].square())
        diagnostics.append(
            dict(
                first_layer=first,
                layers=count_layers,
                rank=rank,
                retained_second_moment_fraction=float(
                    eigenvalues[-rank:].sum() / eigenvalues.sum()
                ),
            )
        )
    return dict(
        group_predictor=torch.stack(predictors),
        group_intercept=torch.stack(intercepts),
        factors=torch.cat(factors),
        noise=torch.cat(noises),
    ), diagnostics
