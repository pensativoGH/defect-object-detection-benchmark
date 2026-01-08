#!/usr/bin/env python3
"""Create combined datasets (real + synthetic) for YOLO training."""

import shutil
from pathlib import Path
import yaml


def create_combined_dataset(
    real_dir: Path,
    synthetic_dir: Path,
    output_dir: Path,
    val_dir: Path,
    test_dir: Path,
    class_names: list
) -> Path:
    """Create a combined dataset from real and synthetic data."""
    output_dir = Path(output_dir)

    # Create output structure
    train_images = output_dir / "train" / "images"
    train_labels = output_dir / "train" / "labels"
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)

    # Copy real training data
    real_images = Path(real_dir) / "images"
    real_labels = Path(real_dir) / "labels"

    copied_real = 0
    if real_images.exists():
        for img in real_images.glob("*.jpg"):
            shutil.copy2(img, train_images / f"real_{img.name}")
            label = real_labels / f"{img.stem}.txt"
            if label.exists():
                shutil.copy2(label, train_labels / f"real_{img.stem}.txt")
            copied_real += 1

    # Copy synthetic data
    synth_images = Path(synthetic_dir) / "images"
    synth_labels = Path(synthetic_dir) / "labels"

    copied_synth = 0
    if synth_images.exists():
        for img in synth_images.glob("*.jpg"):
            shutil.copy2(img, train_images / img.name)
            label = synth_labels / f"{img.stem}.txt"
            if label.exists():
                shutil.copy2(label, train_labels / f"{img.stem}.txt")
            copied_synth += 1

    # Create symlinks for val and test (use original)
    val_link = output_dir / "val"
    test_link = output_dir / "test"

    if val_link.exists():
        shutil.rmtree(val_link)
    if test_link.exists():
        shutil.rmtree(test_link)

    val_link.symlink_to(Path(val_dir).resolve(), target_is_directory=True)
    test_link.symlink_to(Path(test_dir).resolve(), target_is_directory=True)

    # Create data.yaml
    data_yaml = {
        'names': class_names,
        'nc': len(class_names),
        'path': str(output_dir.resolve()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images'
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"Created combined dataset at: {output_dir}")
    print(f"  Real images: {copied_real}")
    print(f"  Synthetic images: {copied_synth}")
    print(f"  Total training images: {copied_real + copied_synth}")

    return yaml_path


if __name__ == "__main__":
    import sys

    base_dir = Path("/Users/pramodsharma/code/synthetic-defect-comparison")
    real_train = base_dir / "data/processed/Cable/train"
    val_dir = base_dir / "data/processed/Cable/val"
    test_dir = base_dir / "data/processed/Cable/inference"
    class_names = ['break', 'thunderbolt']

    # Create combined datasets for each method
    methods = {
        'copypaste': base_dir / "data/synthetic/copypaste",
        'inpainting': base_dir / "data/synthetic/inpainting",
        'controlnet': base_dir / "data/synthetic/controlnet"
    }

    if len(sys.argv) > 1:
        # Process specific method
        method = sys.argv[1]
        if method in methods:
            output = base_dir / f"data/combined/{method}"
            create_combined_dataset(
                real_train, methods[method], output,
                val_dir, test_dir, class_names
            )
    else:
        # Process all methods
        for method, synth_dir in methods.items():
            output = base_dir / f"data/combined/{method}"
            create_combined_dataset(
                real_train, synth_dir, output,
                val_dir, test_dir, class_names
            )
