import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import torch
from torch import nn

from csd.artifacts import checkpoint, digest, load_checkpoint, read, save
from csd.calibration import fit_decoder
from csd.campaign import READERS, make, panel_name, report, required_panels, validate_config
from csd.data import historical_exclusions, normalized_prefix, validate_documents
from csd.evaluation import paired_summary, payload
from csd.model import TRAINED


class ProtocolTests(unittest.TestCase):
    def test_unknown_reader_is_rejected_before_decoder_construction(self):
        with self.assertRaisesRegex(ValueError, "Unknown reader"):
            make("retired_reader", None, None, 11, 10000.0, {})

    def test_historical_export_report_uses_preserved_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = dict(historical_subset_export=True, complete=True, panels=[])
            (root / "historical_report.json").write_text(json.dumps(value))
            with patch("csd.campaign.RESULTS", root), patch("csd.campaign.message"):
                self.assertEqual(report("historical"), value)

    def test_complete_canonical_inventory_and_names(self):
        config = validate_config(json.loads(Path("configs/study.json").read_text()))
        self.assertEqual(len(READERS), len(set(READERS)))
        self.assertTrue(set(TRAINED).issubset(READERS))
        names = []
        for model in config["models"]:
            panels = required_panels(config, model)
            self.assertEqual(
                {p["kind"] for p in panels},
                {"continuation", "recent", "stream", "mechanism", "efficiency"},
            )
            for panel in panels:
                for seed in config["seeds"]:
                    for method in panel["methods"]:
                        name = panel_name("csd", model["name"], panel, method, seed)
                        self.assertLessEqual(len(name), 101)
                        names.append(name)
        self.assertEqual(len(names), len(set(names)))
        wrong = dict(config, methods=config["methods"][:-1])
        with self.assertRaises(ValueError):
            validate_config(wrong)
        self.assertEqual(len(config["models"]), 1)
        self.assertLessEqual(config["max_wall_hours"], 5)
        self.assertEqual(config["tasks"], [])
        self.assertEqual(
            set(config["methods"]) & set(TRAINED),
            {"diagonal", "continued_diagonal", "gaussian", "csd", "headwise", "layerwise"},
        )

    def test_artifacts_immutable_and_checkpoint_content_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            a = save("test", dict(values=[1, 2]), directory)
            self.assertEqual(a, read("test", directory))
            with self.assertRaises(FileExistsError):
                save("test", dict(values=[3]), directory)
            path = Path(directory) / "checkpoint.pt"
            record = checkpoint(path, dict(w=torch.ones(2)))
            torch.testing.assert_close(load_checkpoint(record, "cpu")["w"], torch.ones(2))
            torch.save(dict(w=torch.zeros(2)), path)
            with self.assertRaises(ValueError):
                load_checkpoint(record, "cpu")

    def test_epoch_zero_eligible_and_equal_optimizer_budgets(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.x = nn.Parameter(torch.zeros(2))

        class FakeRunner:
            def data(self, article, cut):
                return {}

            def forward(self, data, model):
                return model.x.square().mean() * 0, (model.x - 1).square()

            def measure(self, articles, cuts, model):
                return [
                    dict(article=a["sha256"], nll=float(model.x.detach().square().mean()), kl=0)
                    for a in articles
                ]

        phases = {p: [dict(sha256=p + str(i)) for i in range(2)] for p in ("fit", "tune")}
        with (
            patch.dict("os.environ", {"CSD_PROGRESS": "0"}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            state, choice, events = fit_decoder(
                Model(), FakeRunner(), phases, [8, 16], [16], 11, 0.01, 3, "test"
            )
        self.assertEqual(choice["selected_epoch"], 0)
        self.assertEqual(choice["optimizer_updates"], 12)
        self.assertEqual(len(events), 3)
        torch.testing.assert_close(state["x"], torch.zeros(2))

    def test_dataset_identity_and_duplicate_guard(self):
        import hashlib

        text = "Test document with enough characters"
        row = dict(sha256=hashlib.sha256(text.encode()).hexdigest(), text=text, phase="fit")
        validate_documents([row])
        with self.assertRaises(ValueError):
            validate_documents([row, dict(row, phase="confirmation")])
        with self.assertRaises(ValueError):
            validate_documents([dict(row, text="changed")])
        tokens, ids, ledger = historical_exclusions()
        self.assertGreater(len(ids), 300)
        self.assertGreater(len(tokens), 100)
        self.assertGreater(len(ledger["prefixes"]), 300)
        self.assertEqual(normalized_prefix(" A\n B "), "a b")

    def test_paired_statistics_reject_missing_or_mismatched_inputs(self):
        rows = [
            dict(
                article=f"{corpus}{i}",
                corpus=corpus,
                seed=seed,
                method=method,
                nll=2.0 + 0.01 * i + (0.05 if method == "gaussian" else 0),
            )
            for corpus in ("wiki", "c4")
            for i in range(3)
            for seed in (11, 22, 33)
            for method in ("csd", "gaussian")
        ]
        result = paired_summary(rows, [11, 22, 33], draws=2000)
        self.assertAlmostEqual(result["comparisons"]["gaussian"]["gain"], 0.05)
        with self.assertRaises(ValueError):
            paired_summary(rows[:-1], [11, 22, 33], draws=100)
        wrong = [dict(r) for r in rows]
        wrong[0]["corpus"] = "other"
        with self.assertRaises(ValueError):
            paired_summary(wrong, [11, 22, 33], draws=100)

    def test_payload_total_includes_retained_values_and_recent_window(self):
        from types import SimpleNamespace
        from csd.model import DiagonalDecoder

        runner = SimpleNamespace(heads=2, dim=16, layers=3, protected=32)
        model = DiagonalDecoder(torch.zeros(3, 2, 1, 16))
        accounting = payload(runner, model, 128)
        self.assertGreater(accounting["retained_value_bytes"], accounting["compressed_key_bytes"])
        self.assertGreater(accounting["decoder_resident_bytes"], 0)
        self.assertLess(accounting["logical_kv_compression"], 2)
        self.assertLess(
            accounting["net_compression_including_decoder"], accounting["logical_kv_compression"]
        )
        self.assertNotEqual(digest(accounting), digest({}))


if __name__ == "__main__":
    unittest.main()
