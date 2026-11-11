import math
import unittest

import numpy as np
from scipy.stats import truncnorm
import torch

from csd.math import infer_latent, latent_terms, positive_gaussian_mean, project_ball, rope
from csd.model import gaussian_estimate
from csd.quantization import RotatedCodec, pack_codes, rotation, unpack_codes


class NumericalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(73)

    def test_packing_and_payload(self):
        for count in (0, 1, 7, 8, 39, 255):
            x = torch.randint(2, (count,))
            packed = pack_codes(x)
            self.assertTrue(torch.equal(unpack_codes(packed, 1, x.shape), x))
            self.assertEqual(packed.numel(), math.ceil(count / 8))
        for x in (torch.tensor([-1]), torch.tensor([2]), torch.ones(3)):
            with self.assertRaises(ValueError):
                pack_codes(x)

    def test_rotation_determinism_and_one_bit_formulas(self):
        r = rotation(16, 11, "cpu")
        torch.testing.assert_close(r.T @ r, torch.eye(16), atol=2e-6, rtol=0)
        self.assertTrue(torch.equal(r, rotation(16, 11, "cpu")))
        key = torch.randn(2, 37, 16)
        for mode in ("unbiased", "optimal"):
            codec = RotatedCodec(16, 2, 11, mode=mode)
            rotated = key @ codec.rot
            amplitude = (
                rotated.square().sum(-1, keepdim=True) / rotated.abs().sum(-1, keepdim=True)
                if mode == "unbiased"
                else rotated.abs().mean(-1, keepdim=True)
            )
            cache = codec.encode(key)
            torch.testing.assert_close(
                codec.vector(cache), rotated.sign() * amplitude.half().float()
            )
            self.assertEqual(cache.nbytes(), math.ceil(key.numel() / 8) + 2 * 2 * 37)
            torch.testing.assert_close(
                codec.decode(codec.encode(torch.zeros_like(key))), torch.zeros_like(key)
            )

    def test_rope_matches_native_operation_order(self):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        from transformers import Qwen2Config

        x = torch.randn(2, 3, 128)
        positions = torch.tensor([0, 4095, 16383])
        for theta in (10000.0, 1000000.0):
            module = Qwen2RotaryEmbedding(
                Qwen2Config(hidden_size=256, num_attention_heads=2, rope_theta=theta)
            )
            cosine, sine = module(x[None], positions[None])
            half = torch.cat([-x[..., 64:], x[..., :64]], -1)
            expected = x * cosine[0] + half * sine[0]
            torch.testing.assert_close(rope(x, positions, theta), expected, rtol=0, atol=0)
            torch.testing.assert_close(
                rope(rope(x, positions, theta), positions, theta, True), x, atol=8e-7, rtol=2e-6
            )

    def test_gaussian_tail_means_and_gradients(self):
        x = torch.tensor([-1000.0, -100.0, -20.0, -9.0, -5.0, 0.0, 5.0, 100.0], requires_grad=True)
        value = positive_gaussian_mean(x, torch.ones_like(x), torch.ones_like(x))
        with np.errstate(invalid="ignore"):
            expected = np.array(
                [truncnorm.mean(-float(v), np.inf, loc=float(v)) for v in x.detach()]
            )
        np.testing.assert_allclose(value.detach().numpy(), expected, rtol=2e-5, atol=2e-7)
        value.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        opposite = positive_gaussian_mean(-x.detach(), torch.ones_like(x), -torch.ones_like(x))
        torch.testing.assert_close(opposite, -value.detach())

    def test_probit_root_implicit_gradient_and_hessian(self):
        a = (torch.randn(2, 5, dtype=torch.float64) * 0.6).requires_grad_()
        b = (torch.randn(2, 5, 3, dtype=torch.float64) * 0.3).requires_grad_()
        z = infer_latent(a, b)
        gradient, hessian = latent_terms(z, a, b)
        self.assertLess(float(gradient.detach().abs().max()), 1e-9)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(hessian.detach()).min()), 1)
        self.assertTrue(torch.autograd.gradcheck(infer_latent, (a, b), atol=3e-5, rtol=3e-4))
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda a, b: infer_latent(a, b, 1), (a, b), atol=3e-5, rtol=3e-4
            )
        )

    def test_gaussian_control_dense_reference(self):
        mean = torch.randn(2, 2, 3, 4, dtype=torch.float64)
        noise, qnoise = torch.rand_like(mean) + 0.1, torch.rand_like(mean) + 0.1
        basis = torch.randn(*mean.shape, 3, dtype=torch.float64) * 0.2
        observed = torch.randn_like(mean)
        output = gaussian_estimate(mean, noise, basis, observed, qnoise)
        for token in range(3):
            u = basis[:, :, token].reshape(16, 3)
            covariance = torch.diag(noise[:, :, token].flatten()) + u @ u.T
            m, y = mean[:, :, token].flatten(), observed[:, :, token].flatten()
            expected = m + covariance @ torch.linalg.solve(
                covariance + torch.diag(qnoise[:, :, token].flatten()), y - m
            )
            torch.testing.assert_close(output[:, :, token].flatten(), expected)

    def test_projection_metadata_ball(self):
        standard, desired = torch.randn(12, 16), torch.randn(12, 16) * 4
        result = project_ball(desired, standard)
        violation = result.square().sum(-1) - (result * standard).sum(-1)
        self.assertLess(float(violation.max()), 5e-6)


if __name__ == "__main__":
    unittest.main()
