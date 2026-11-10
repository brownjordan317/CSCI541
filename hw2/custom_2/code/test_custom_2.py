"""Fast checks: python -m unittest custom_2.code.test_custom_2."""

import unittest

import torch

from custom_2.code.train_custom_2 import (
    CONFIGS, CompactCNN, CustomCNN, robust_augment, stress_views,
)
from custom_2.code.train_tiny_cosine import learning_rate
from custom_2.code.lr_control import PlateauController
from custom_2.code.optimize_score import CONFIGS as SCORE_CONFIGS, ScoreCNN


class Custom2Tests(unittest.TestCase):
    def test_score_architectures_have_direct_five_logit_heads(self):
        for config in SCORE_CONFIGS:
            model = ScoreCNN(config.widths, config.blocks_per_stage)
            count = sum(p.numel() for p in model.parameters())
            self.assertLess(count, 511957)
            self.assertEqual(sum(isinstance(m, torch.nn.Linear) for m in model.modules()), 1)
            result = model(torch.rand(2, 3, 64, 64))
            self.assertEqual(tuple(result.shape), (2, 5))
            torch.nn.CrossEntropyLoss()(result, torch.tensor([0, 4])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()), config.name)

    def test_plateau_reduces_lr_before_stopping(self):
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=1e-3)
        controller = PlateauController(optimizer, lr_patience=0, stop_patience=3)
        self.assertFalse(controller.update(0.8, trained_lr=1e-3))
        for _ in range(7):
            used_lr = optimizer.param_groups[0]["lr"]
            self.assertGreater(used_lr, 1e-5)
            self.assertFalse(controller.update(0.8, trained_lr=used_lr))
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)
        self.assertEqual(controller.floor_epochs, 0)
        # A full fresh patience window is needed after actual minimum-LR training.
        for _ in range(3):
            self.assertFalse(controller.update(0.8, trained_lr=1e-5))
        self.assertTrue(controller.update(0.8, trained_lr=1e-5))

    def test_min_lr_improvement_resets_stop_patience(self):
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=1e-5)
        controller = PlateauController(optimizer, stop_patience=2)
        self.assertFalse(controller.update(0.8, trained_lr=1e-5))
        self.assertFalse(controller.update(0.8, trained_lr=1e-5))
        self.assertFalse(controller.update(0.805, trained_lr=1e-5))
        self.assertEqual(controller.floor_stale, 0)
        self.assertFalse(controller.update(0.805, trained_lr=1e-5))
        self.assertTrue(controller.update(0.805, trained_lr=1e-5))

    def test_cosine_schedule(self):
        rates = [learning_rate(epoch, 80) for epoch in range(1, 81)]
        self.assertAlmostEqual(rates[0], 1e-3)
        self.assertAlmostEqual(rates[-1], 1e-5)
        self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))
        self.assertTrue(all(1e-5 <= value <= 1e-3 for value in rates))
        self.assertAlmostEqual(learning_rate(81, 80), 1e-5)
        self.assertAlmostEqual(learning_rate(160, 80), 1e-5)

    def test_original_widths_reproduce_original_model(self):
        torch.manual_seed(7)
        original = CustomCNN().eval()
        torch.manual_seed(7)
        reconstructed = CompactCNN((32, 64, 128, 192)).eval()
        self.assertEqual(list(original.state_dict()), list(reconstructed.state_dict()))
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, reconstructed.state_dict()[key]), key)
        x = torch.rand(2, 3, 64, 64)
        self.assertTrue(torch.equal(original(x), reconstructed(x)))

    def test_budget_shapes_and_backward(self):
        for widths in {(24, 48, 96, 144), (16, 32, 64, 96)}:
            model = CompactCNN(widths)
            floats = sum(t.numel() for t in model.state_dict().values() if t.is_floating_point())
            self.assertLess(floats, 3_000_000)
            output = model(torch.rand(2, 3, 64, 64))
            self.assertEqual(tuple(output.shape), (2, 5))
            torch.nn.CrossEntropyLoss()(output, torch.tensor([0, 1])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))

    def test_clean_mixture_and_bounds(self):
        x = torch.rand(64, 3, 64, 64)
        torch.manual_seed(11)
        flipped = x.clone()
        flip = torch.rand(len(x)) < 0.5
        flipped[flip] = flipped[flip].flip(-1)
        clean = torch.rand(len(x)) < 0.2
        self.assertTrue(clean.any())
        torch.manual_seed(11)
        result = robust_augment(x.clone())
        self.assertTrue(torch.equal(result[clean], flipped[clean]))
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreaterEqual(result.min().item(), 0)
        self.assertLessEqual(result.max().item(), 1)

    def test_stress_suite_deterministic_without_rng_side_effect(self):
        x = torch.rand(2, 3, 64, 64)
        state = torch.random.get_rng_state().clone()
        first = stress_views(x)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        second = stress_views(x)
        self.assertEqual(len(first), 12)
        for name in first:
            self.assertTrue(torch.equal(first[name], second[name]), name)
            self.assertTrue(torch.isfinite(first[name]).all(), name)
            self.assertGreaterEqual(first[name].min().item(), 0)
            self.assertLessEqual(first[name].max().item(), 1)

    def test_controlled_configs(self):
        self.assertEqual(len(CONFIGS), 4)
        self.assertEqual(CONFIGS[0].widths, CONFIGS[1].widths)
        self.assertEqual(CONFIGS[1].lr, CONFIGS[2].lr)
        self.assertEqual(CONFIGS[1].widths, CONFIGS[3].widths)


if __name__ == "__main__":
    unittest.main()
