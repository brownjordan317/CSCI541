"""Create random Copybook augmentations from the original dataset.

Usage: python hw1/augment_dataset.py 3 [--seed 42]
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

from sample_mlp import load_copybook, StrokeWidthJitter
from image_augmentors.color_jitter import ColorJitter
from image_augmentors.elastic_distortion import ElasticDistortion
from image_augmentors.gaussian_filter import GaussianFilter
from image_augmentors.image_filtering import ImageFiltering
from image_augmentors.rotate_image import RotateImage
from image_augmentors.scale_image import ScaleImage
from image_augmentors.translate_image import TranslateImage
from image_augmentors.warp_image import WarpImage

# Small handwriting variations on a 64x64 canvas.
MAX_ROTATION_DEGREES = 20.0
MAX_TRANSLATION_PIXELS = 15
MAX_SHEAR = 0.05
MAX_SCALE_CHANGE = 0.15
MAX_ELASTIC_PIXELS = 5
MIN_INK_FRACTION = 0.5
INK_CUTOFF = 245
MAX_ATTEMPTS = 50

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "original_dataset"
DESTINATION = ROOT / "augmented_dataset"


def random_transform(name, image, rng):
    """Apply one random transform to a uint8 grayscale image."""
    if name == "rotate":
        angle = rng.uniform(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES)
        return RotateImage(float(angle))(image)
    if name == "translate":
        limit = MAX_TRANSLATION_PIXELS
        dx, dy = rng.integers(-limit, limit + 1, size=2)
        return TranslateImage(int(dx), int(dy))(image)
    if name == "warp":
        shear_x, shear_y = rng.uniform(-MAX_SHEAR, MAX_SHEAR, size=2)
        matrix = np.float32([[1, shear_x, -32 * shear_x],
                             [shear_y, 1, -32 * shear_y]])
        return WarpImage(matrix)(image)
    if name == "scale":
        scale = 1.0 + rng.uniform(-MAX_SCALE_CHANGE, MAX_SCALE_CHANGE)
        return ScaleImage(float(scale))(image)
    if name == "elastic":
        return ElasticDistortion(rng, MAX_ELASTIC_PIXELS)(image)
    if name == "stroke_width":
        direction = str(rng.choice(["thin", "thick"]))
        strength = float(rng.uniform(0.35, 0.65))
        return StrokeWidthJitter(direction, strength)(image)
    if name == "color_jitter":
        brightness = float(rng.uniform(0.85, 1.15))
        contrast = float(rng.uniform(0.85, 1.15))
        return ColorJitter(brightness, contrast)(image)
    if name == "gaussian":
        return GaussianFilter(3, 3, float(rng.uniform(0.3, 0.9))).apply_filter(
            image
        )
    if name == "median":
        return ImageFiltering("median", kernel_size=(3, 3)).apply(image)
    raise ValueError(f"unknown transform: {name}")


def augment(image, rng):
    """Apply one to three distinct augmentations in random order."""
    choices = ("rotate", "translate", "warp", "scale", "elastic",
               "gaussian", "median", "color_jitter", "stroke_width")
    count = int(rng.integers(1, 4))
    for name in rng.choice(choices, size=count, replace=False):
        image = random_transform(name, image, rng)
    return image


def has_enough_ink(candidate, original):
    """Reject blank or nearly blank images relative to their source."""
    source_ink = np.count_nonzero(original < INK_CUTOFF)
    candidate_ink = np.count_nonzero(candidate < INK_CUTOFF)
    return candidate_ink >= max(1, source_ink * MIN_INK_FRACTION)


def generate(times, seed=None):
    if times < 1:
        raise ValueError("times must be at least 1")
    rng = np.random.default_rng(seed)
    images, labels = load_copybook(str(SOURCE))
    with (SOURCE / "labels.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if isinstance(metadata, dict):
        metadata = metadata.get("labels") or metadata.get("shots")
    if len(metadata) != len(images) or len(labels) != len(images):
        raise ValueError("labels.json and image count differ")

    shots = DESTINATION / "shots"
    shots.mkdir(parents=True, exist_ok=True)
    output_labels = []
    rejected = 0
    with tqdm(
        total=len(images) * times, desc="Augmenting", unit="image"
    ) as progress:
        for index, (image, item) in enumerate(zip(images, metadata)):
            original_name = f"{index:05d}.png"
            original_path = SOURCE / "shots" / item["file"]
            shutil.copyfile(original_path, shots / original_name)
            output_labels.append({**item, "file": original_name})

            pixels = np.rint(image * 255).astype(np.uint8)
            for copy_index in range(times):
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    augmented = augment(pixels.copy(), rng)
                    if has_enough_ink(augmented, pixels):
                        break
                    rejected += 1
                else:
                    raise RuntimeError(
                        f"Could not make a valid augmentation for "
                        f"{item['file']} copy {copy_index + 1} after "
                        f"{MAX_ATTEMPTS} attempts"
                    )
                filename = f"{index:05d}_aug{copy_index + 1:03d}.png"
                Image.fromarray(
                    np.asarray(augmented, dtype=np.uint8), "L"
                ).save(shots / filename)
                output_labels.append({
                    **item, "file": filename,
                    "augmented_from": item["file"],
                })
                progress.update(1)
        progress.set_postfix(rejected=rejected)

    if len(output_labels) != len(images) * (times + 1):
        raise RuntimeError("generated label count does not match image count")
    if any(not (shots / item["file"]).is_file() for item in output_labels):
        raise RuntimeError("a generated image is missing")

    labels_file = DESTINATION / "labels.json"
    temporary_file = DESTINATION / "labels.json.tmp"
    with temporary_file.open("w", encoding="utf-8") as stream:
        json.dump(output_labels, stream, ensure_ascii=False, indent=2)
    temporary_file.replace(labels_file)
    print(
        f"Saved {len(output_labels)} images and {labels_file} "
        f"({rejected} rejected attempts)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("times", type=int, help="augmented copies per image")
    parser.add_argument(
        "--seed", type=int, help="random seed for repeatability"
    )
    args = parser.parse_args()
    generate(args.times, args.seed)


if __name__ == "__main__":
    main()
