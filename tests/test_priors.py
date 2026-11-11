import contextlib
import io
import os
from unittest.mock import patch
import unittest

import torch

from csd.math import rope
from csd.priors import fit_structured_prior


class SyntheticSource:
    layers, heads, dim, theta = 4, 1, 8, 10000.0

    def data(self, article, cut):
        generator = torch.Generator().manual_seed(int(article["sha256"]) + 611)
        keys = [torch.randn(1, 1, cut - 1, 8, generator=generator) for _ in range(4)]
        values = [torch.randn(1, 1, cut - 1, 8, generator=generator) for _ in range(4)]
        return dict(keys=keys, values=values)


class PriorTests(unittest.TestCase):
    @patch.dict(os.environ, {"CSD_PROGRESS": "0"})
    def test_partial_group_dimensions_and_cross_fitted_variance(self):
        runner = SyntheticSource()
        articles = [dict(sha256=str(i)) for i in range(2)]
        with contextlib.redirect_stdout(io.StringIO()):
            prior, diagnostics = fit_structured_prior(
                runner, articles, 17, group_size=3, rank_per_layer=2
            )
        self.assertEqual(prior["group_predictor"].shape, (2, 24, 24))
        self.assertEqual(prior["factors"].shape, (4, 1, 6, 8))
        self.assertEqual([d["layers"] for d in diagnostics], [3, 1])
        self.assertTrue(all(torch.isfinite(v).all() for v in prior.values()))
        self.assertTrue(bool((prior["noise"] > 0).all()))
        self.assertEqual(int(torch.count_nonzero(prior["factors"][3, :, 2:])), 0)
        # With independent random keys, an overparameterized fit can memorize noise.
        # Article-held-out residual variance must not collapse to its in-fit error.
        errors = []
        for article in articles:
            data = runner.data(article, 17)
            x = torch.cat([v[0, 0] for v in data["values"][:3]], -1)
            y = torch.cat(
                [rope(k[0], torch.arange(16), 10000.0, inverse=True)[0] for k in data["keys"][:3]],
                -1,
            )
            errors.append(
                (y - x @ prior["group_predictor"][0] - prior["group_intercept"][0]).square().mean()
            )
        in_fit = torch.stack(errors).mean()
        fitted_marginal = prior["factors"][:3].square().sum(-2).mean() + prior["noise"][:3].mean()
        self.assertGreater(float(fitted_marginal), 2 * float(in_fit))

    def test_cross_fitting_rejects_one_article(self):
        with self.assertRaises(ValueError):
            fit_structured_prior(SyntheticSource(), [dict(sha256="0")], 17)


if __name__ == "__main__":
    unittest.main()
