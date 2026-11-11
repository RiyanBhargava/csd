import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import torch

from csd import artifacts
from csd.campaign import READERS, calibrate, continuation_panel, restore
from csd.model import ABLATIONS, TRAINED, CooperativeDecoder
from csd.tasks import greedy
from tests.test_model import fixture


class IntegrationTests(unittest.TestCase):
    @patch.dict(os.environ, {"CSD_PROGRESS": "0"})
    def test_calibration_selection_restore_and_resume_every_family(self):
        class FakeRunner:
            theta, identity = 10000.0, dict(model="fake")

            def data(self, article, cut):
                return {}

            def forward(self, data, decoder):
                loss = sum(
                    (p - 0.2).square().mean() for p in decoder.parameters() if p.requires_grad
                )
                return loss * 0.1, loss.reshape(1)

            def measure(self, articles, cuts, decoder):
                value = float(sum(p.detach().square().mean() for p in decoder.parameters()))
                return [dict(article=a["sha256"], corpus="wiki", nll=value, kl=0) for a in articles]

        config = json.loads(Path("configs/study.json").read_text())
        config.update(
            methods=list(READERS),
            epochs=1,
            lrs=[0.001, 0.003],
            linear_lrs=[0.0001, 0.001, 0.003],
            train_cuts=[8],
            tune_cuts=[8],
        )
        prior = fixture()
        documents = [
            dict(sha256=f"{phase}{i}", phase=phase) for phase in ("fit", "tune") for i in range(2)
        ]
        plan = dict(sha256="integration-plan")
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(patch("csd.campaign.RESULTS", root))
            stack.enter_context(patch("csd.campaign.CACHE", root / "cache"))
            stack.enter_context(patch("csd.campaign.read", lambda name: artifacts.read(name, root)))
            stack.enter_context(patch("csd.campaign.save", lambda n, v: artifacts.save(n, v, root)))
            stack.enter_context(
                patch(
                    "csd.campaign.load_checkpoint",
                    lambda record: artifacts.load_checkpoint(record, "cpu"),
                )
            )
            stack.enter_context(patch("csd.campaign.prepare_prior", return_value=prior))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            artifacts.save("test_fake_prior", dict(plan_sha256=plan["sha256"]), root)
            calibrate(FakeRunner(), documents, config, "test", "fake", plan)
            for seed in config["seeds"]:
                for method in TRAINED:
                    decoder = restore(
                        method, FakeRunner(), prior, config, "test", "fake", seed, plan
                    )
                    record = artifacts.read(f"test_fake_{method}_{seed}", root)
                    expected_rates = config["linear_lrs"] if method == "linear" else config["lrs"]
                    self.assertEqual(len(record["trials"]), len(expected_rates))
                    self.assertTrue(all(t["optimizer_updates"] == 2 for t in record["trials"]))
                    self.assertEqual(
                        record["selected"]["tune_nll"], min(t["tune_nll"] for t in record["trials"])
                    )
                    if method in ("csd", *ABLATIONS):
                        self.assertEqual(decoder.kind, method)
                    self.assertNotIn("factors", decoder.state_dict())
            saved = {p.name: p.read_bytes() for p in root.glob("*.json.gz")}
            calibrate(FakeRunner(), documents, config, "test", "fake", plan)
            self.assertEqual(saved, {p.name: p.read_bytes() for p in root.glob("*.json.gz")})

    @patch.dict(os.environ, {"CSD_PROGRESS": "0"})
    def test_document_batches_resume_and_reuse_deterministic_references(self):
        class FakeRunner:
            identity = dict(model="fake")

            def measure(self, documents, cuts, decoder):
                return [
                    dict(article=a["sha256"], corpus=a["corpus"], nll=2.0, kl=0.0)
                    for a in documents
                ]

        documents = [dict(sha256=str(i), corpus="wiki") for i in range(5)]
        config = dict(seeds=[11, 22, 33], evaluation_batch_documents=2)
        panel = dict(kind="continuation", phase="confirmation", context=16, methods=["bf16", "csd"])
        plan = dict(sha256="batch-plan")
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(patch("csd.campaign.RESULTS", root))
            stack.enter_context(patch("csd.campaign.read", lambda n: artifacts.read(n, root)))
            stack.enter_context(patch("csd.campaign.save", lambda n, v: artifacts.save(n, v, root)))
            restore_mock = stack.enter_context(patch("csd.campaign.restore", return_value=None))
            continuation_panel(FakeRunner(), documents, config, "test", "fake", plan, {}, panel)
            self.assertEqual(
                restore_mock.call_count, 12
            )  # BF16 once + CSD three seeds, three chunks
            record = artifacts.read("test_fake_continuation_confirmation_16_csd_33", root)
            self.assertEqual(
                [r["article"] for r in record["observations"]], list(map(str, range(5)))
            )
            saved = {p.name: p.read_bytes() for p in root.glob("*.json.gz")}
            continuation_panel(FakeRunner(), documents, config, "test", "fake", plan, {}, panel)
            self.assertEqual(saved, {p.name: p.read_bytes() for p in root.glob("*.json.gz")})

    def test_generation_prepares_all_past_layers_before_first_answer(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(15)
        config = Qwen2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            rope_theta=10000.0,
            max_position_embeddings=256,
            sliding_window=None,
        )
        model = Qwen2ForCausalLM(config).eval().requires_grad_(False)
        decoder = CooperativeDecoder(fixture(), torch.zeros(4, 2, 16), theta=10000.0)
        bindings, reads = [], []
        original_bind, original_read = decoder.bind_cache, decoder.with_values

        def bind(keys, values, protected):
            bindings.append(([k.shape[-2] for k in keys], protected))
            original_bind(keys, values, protected)

        def read(key, values, layer, positions):
            reads.append((layer, key.shape[-2]))
            return original_read(key, values, layer, positions)

        with (
            patch.object(decoder, "bind_cache", side_effect=bind),
            patch.object(decoder, "with_values", side_effect=read),
        ):
            generated = greedy(model, list(range(1, 65)), decoder, 2, set(), 16)
        self.assertEqual(len(generated), 2)
        self.assertEqual(bindings, [([63] * 4, 15), ([64] * 4, 15)])
        self.assertEqual(reads, [(i, n) for n in (48, 49) for i in range(4)])
        self.assertIsNone(decoder._bound)

    def test_all_suffix_targets_match_independent_full_context_reference(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from csd.runner import Runner, prefill

        torch.manual_seed(194)
        config = Qwen2Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            rope_theta=10000.0,
            sliding_window=None,
        )
        model = Qwen2ForCausalLM(config).eval().requires_grad_(False)
        runner = Runner.__new__(Runner)
        runner.model, runner.protected = model, 8
        tokens = list(range(1, 49))
        cut, suffix = 32, 16
        with torch.no_grad():
            reference = (
                model(torch.tensor([tokens[:-1]]), use_cache=False).logits[0].log_softmax(-1)
            )
            cache = prefill(model, tokens[: cut - 1])
            data = dict(
                keys=cache.key_cache,
                values=cache.value_cache,
                tokens=tokens,
                prefix=cut,
                logp=reference[cut - 1 :],
            )
            kl, nll = runner.forward(data)
        expected = -reference[cut - 1 :].gather(-1, torch.tensor(tokens[cut:])[:, None]).squeeze(-1)
        self.assertEqual(len(nll), suffix)
        torch.testing.assert_close(nll, expected, rtol=1e-6, atol=1e-6)
        self.assertLess(abs(float(kl)), 2e-7)

    def test_local_reader_resets_once_per_generated_sequence(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM

        class Spy:
            starts = 0

            def start_sequence(self):
                self.starts += 1

            def pair(self, key, value, layer):
                return key, value

        config = Qwen2Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            sliding_window=None,
        )
        model = Qwen2ForCausalLM(config).eval().requires_grad_(False)
        reader = Spy()
        greedy(model, list(range(1, 49)), reader, 3, set(), 16)
        self.assertEqual(reader.starts, 1)


if __name__ == "__main__":
    unittest.main()
