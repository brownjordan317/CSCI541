"""Validation-driven LR reduction; stopping is gated on training at min LR."""

import math

import torch


class PlateauController:
    def __init__(self, optimizer, minimum=1e-5, lr_patience=4,
                 stop_patience=15, min_delta=0.001):
        self.optimizer = optimizer
        self.minimum = minimum
        self.stop_patience = stop_patience
        self.min_delta = min_delta
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=lr_patience,
            threshold=min_delta, threshold_mode="abs", min_lr=minimum,
        )
        self.floor_reference = None
        self.floor_stale = 0
        self.floor_epochs = 0

    def update(self, score, trained_lr):
        # Use the LR actually used for this epoch, not the next LR produced
        # by scheduler.step(). Merely reaching the floor after validation
        # does not count as having trained at that floor.
        trained_at_floor = math.isclose(trained_lr, self.minimum, rel_tol=1e-8, abs_tol=1e-12)
        if trained_at_floor:
            self.floor_epochs += 1
            if self.floor_reference is None or score > self.floor_reference + self.min_delta:
                self.floor_reference = score
                self.floor_stale = 0
            else:
                self.floor_stale += 1
        else:
            self.floor_reference = None
            self.floor_stale = 0
        self.scheduler.step(score)
        return trained_at_floor and self.floor_stale >= self.stop_patience
