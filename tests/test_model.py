from unittest.mock import patch
import unittest

import torch

from csd.model import ABLATIONS, CooperativeDecoder, DiagonalDecoder
from csd.quantization import RotatedCodec
from csd.runner import compressed_pair


def fixture(layers=4, heads=2, dim=16, group=3):
    torch.manual_seed(101)
    width = group * heads * dim
    return dict(
        centers=torch.randn(layers, heads, 1, dim) * 0.03,
        factors=torch.randn(layers, heads, 4, dim) * 0.1,
        noise=torch.full((layers, heads), 0.2),
        group_predictor=torch.randn((layers + group - 1) // group, width, width) * 0.04,
        group_intercept=torch.randn((layers + group - 1) // group, width) * 0.02,
    )


class DecoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.prior = fixture()
        self.diagonal = torch.rand(4, 2, 16) * 0.4
        self.keys = [torch.randn(1, 2, 21, 16, requires_grad=True) for _ in range(4)]
        self.values = [torch.randn_like(k, requires_grad=True) for k in self.keys]

    def decoder(self, kind="csd", **kwargs):
        return CooperativeDecoder(self.prior, self.diagonal, kind=kind, token_chunk=7, **kwargs)

    def decode(self, model):
        model.bind_cache(self.keys, self.values, 4)
        model.start_sequence()
        return torch.stack(list(model.prepared.values()))

    def test_partial_group_all_components_and_gradients(self):
        for kind in ("csd", "gaussian", "value_only", *ABLATIONS):
            model = self.decoder(kind)
            output = self.decode(model)
            self.assertEqual(output.shape, (4, 2, 17, 16))
            output.square().mean().backward()
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, (kind, name))
                    self.assertTrue(torch.isfinite(p.grad).all(), (kind, name))
            self.assertTrue(all(t.grad is None for t in [*self.keys, *self.values]))

    def test_beta_zero_reconstructs_shared_diagonal_anchor(self):
        model = self.decoder()
        model.blend.data.zero_()
        output = self.decode(model)
        control = DiagonalDecoder(self.prior["centers"])
        control.diagonal.data.copy_(self.diagonal)
        for layer in range(4):
            torch.testing.assert_close(
                output[layer], control(self.keys[layer][0, :, :17], layer), atol=2e-6, rtol=2e-6
            )

    def test_checkpoint_outputs_and_gradients_equal_after_release(self):
        for kind in ("csd", "gaussian"):
            left, right = self.decoder(kind), self.decoder(kind, checkpoint_chunks=False)
            right.load_state_dict(left.state_dict())
            a, b = self.decode(left), self.decode(right)
            left.release_cache()
            right.release_cache()
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            a.square().sum().backward()
            b.square().sum().backward()
            for p, q in zip(left.parameters(), right.parameters(), strict=True):
                if p.requires_grad:
                    torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)

    def test_no_future_token_or_other_depth_group_access(self):
        model = self.decoder()
        with torch.no_grad():
            original = self.decode(model).clone()
            for v in self.values:
                v[..., 17:, :] *= 100
            torch.testing.assert_close(original, self.decode(model), rtol=0, atol=0)
            self.values[3][..., :17, :] *= 2
            changed = self.decode(model)
            torch.testing.assert_close(original[:3], changed[:3], rtol=0, atol=0)
            self.assertGreater(float((original[3] - changed[3]).abs().max()), 1e-4)

    def test_value_only_never_reads_key_contents(self):
        model = self.decoder("value_only")
        with patch.object(RotatedCodec, "encode", side_effect=AssertionError("key read")):
            original = self.decode(model)
            with torch.no_grad():
                for k in self.keys:
                    k.fill_(float("nan"))
            torch.testing.assert_close(original, self.decode(model), rtol=0, atol=0)
        self.assertNotIn("factors", dict(model.named_buffers()))

    def test_no_correlation_preserves_covariance_marginals(self):
        model = self.decoder("no_correlation")
        self.assertFalse(model.coupling_logit.requires_grad)
        a = self.decode(model)
        model.coupling_logit.data.fill_(7)
        b = self.decode(model)
        torch.testing.assert_close(a, b, rtol=3e-6, atol=3e-6)

    def test_safe_binding_and_no_inherited_local_fallback(self):
        model = self.decoder()
        with self.assertRaises(ValueError):
            model.start_sequence()
        with self.assertRaises(ValueError):
            model(self.keys[0][0], 0)
        with self.assertRaises(ValueError):
            model.bind_cache(self.keys[:-1], self.values, 4)
        self.decode(model)
        with self.assertRaises(ValueError):
            model.with_values(self.keys[0][0], self.values[0][0], 0)

    def test_exact_recent_window(self):
        model = self.decoder()
        self.decode(model)
        key, value = compressed_pair(self.keys[0], self.values[0], model, 0, 4)
        torch.testing.assert_close(key[..., 17:, :], self.keys[0][..., 17:, :], rtol=0, atol=0)
        torch.testing.assert_close(value, self.values[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
