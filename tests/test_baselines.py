import unittest
from unittest.mock import Mock

import torch

from csd.baselines import Eden, Hqq


class BaselineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(219)
        torch.set_num_threads(2)

    def test_local_quantizer_reuses_prefix_and_quantizes_appended_tokens(self):
        reader = Hqq.__new__(Hqq)
        reader.start_sequence()
        reader.quantize = Mock(side_effect=lambda key: key + 1)
        key, values = torch.randn(2, 5, 64), torch.randn(2, 5, 64)
        decoded, retained = reader.pair(key, values)
        torch.testing.assert_close(decoded, key + 1)
        self.assertIs(retained, values)
        reader.pair(key, values)
        self.assertEqual(reader.quantize.call_count, 1)
        extended = torch.cat([key, torch.randn(2, 2, 64)], dim=1)
        decoded, _ = reader.pair(extended, torch.randn_like(extended))
        torch.testing.assert_close(decoded, extended + 1)
        self.assertEqual(reader.quantize.call_args.args[0].shape[1], 2)

    def test_published_rotated_scale_controls(self):
        centers = torch.randn(2, 2, 1, 16) * 0.1
        key = torch.randn(2, 27, 16)
        for mode in ("unbiased", "mse", "matched_norm"):
            reader = Eden(centers, 11, mode)
            codec = reader.codecs[0]
            y = (key - centers[0]) @ codec.rot
            if mode == "unbiased":
                amplitude = y.square().sum(-1, keepdim=True) / y.abs().sum(-1, keepdim=True)
            elif mode == "mse":
                amplitude = y.abs().mean(-1, keepdim=True)
            else:
                amplitude = (key - centers[0]).norm(dim=-1, keepdim=True) / 4
            expected = (y.sign() * amplitude.half().float()) @ codec.rot.transpose(
                -1, -2
            ) + centers[0]
            torch.testing.assert_close(reader(key, 0), expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
