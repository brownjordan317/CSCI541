"""Local tests: python -m unittest p3_best.test_model -v (from hw2/)."""

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .model import MiniGAPCNN
from .training_utils import PlateauController, robust_augment, validation_views

HERE = Path(__file__).resolve().parent


class BestModelTests(unittest.TestCase):
    def test_copied_artifact_hashes(self):
        manifest = json.loads((HERE / "results/provenance.json").read_text())
        for record in manifest["artifacts"]:
            path = HERE / Path(record["copy"]).relative_to("p3_best")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])

    def test_model_matches_preserved_onnx(self):
        checkpoint = torch.load(HERE / "checkpoints/best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["epoch"], 79)
        model = MiniGAPCNN().eval()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 174933)
        torch.manual_seed(17)
        images = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            expected = model(images).numpy()
        session = ort.InferenceSession(str(HERE / "p3_best.onnx"), providers=["CPUExecutionProvider"])
        actual = np.concatenate([session.run(None, {"image": image[None].numpy()})[0] for image in images])
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def test_training_data_utilities(self):
        images = torch.rand(8, 3, 64, 64)
        result = robust_augment(images.clone())
        self.assertEqual(result.shape, images.shape)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreaterEqual(result.min().item(), 0)
        self.assertLessEqual(result.max().item(), 1)
        self.assertEqual(len(validation_views(images)), 13)

    def test_stopping_is_gated_on_training_at_minimum_lr(self):
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=1e-3)
        controller = PlateauController(optimizer, lr_patience=0, stop_patience=2)
        self.assertFalse(controller.update(0.8, trained_lr=1e-3))
        for _ in range(7):
            lr = optimizer.param_groups[0]["lr"]
            self.assertGreater(lr, 1e-5)
            self.assertFalse(controller.update(0.8, trained_lr=lr))
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)
        self.assertEqual(controller.floor_epochs, 0)
        self.assertFalse(controller.update(0.8, trained_lr=1e-5))
        self.assertFalse(controller.update(0.8, trained_lr=1e-5))
        self.assertTrue(controller.update(0.8, trained_lr=1e-5))


if __name__ == "__main__":
    unittest.main()
