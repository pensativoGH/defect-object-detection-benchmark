#!/usr/bin/env python3
"""Train Qwen2-VL for defect detection using LoRA.

Usage:
    # Train on real data only
    python scripts/train_qwen_vl.py --data real --output-dir models/qwen_vl/real

    # Train on real + synthetic data
    python scripts/train_qwen_vl.py --data combined --output-dir models/qwen_vl/combined
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.qwen_vl_trainer import QwenVLTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Qwen2-VL for defect detection")
    parser.add_argument(
        "--data",
        type=str,
        choices=["real", "combined_ti", "combined_lora", "combined_copypaste"],
        default="real",
        help="Data source: real only or combined with synthetic",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/qwen_vl.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent

    # Determine data directories based on --data argument
    if args.data == "real":
        images_dir = project_root / "data" / "processed" / "Cable" / "train" / "images"
        labels_dir = project_root / "data" / "processed" / "Cable" / "train" / "labels"
        default_output = "models/qwen_vl/real"
    elif args.data == "combined_ti":
        images_dir = project_root / "data" / "combined" / "textual_inversion" / "train" / "images"
        labels_dir = project_root / "data" / "combined" / "textual_inversion" / "train" / "labels"
        default_output = "models/qwen_vl/combined_ti"
    elif args.data == "combined_lora":
        images_dir = project_root / "data" / "combined" / "lora" / "train" / "images"
        labels_dir = project_root / "data" / "combined" / "lora" / "train" / "labels"
        default_output = "models/qwen_vl/combined_lora"
    elif args.data == "combined_copypaste":
        images_dir = project_root / "data" / "combined" / "copy_paste" / "train" / "images"
        labels_dir = project_root / "data" / "combined" / "copy_paste" / "train" / "labels"
        default_output = "models/qwen_vl/combined_copypaste"

    output_dir = args.output_dir or default_output
    config_path = project_root / args.config

    print(f"Training Qwen2-VL with data: {args.data}")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"Output: {output_dir}")

    # Check directories exist
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)
    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        sys.exit(1)

    # Create trainer
    trainer = QwenVLTrainer(config_path)

    # Setup model
    print("\nLoading model and applying LoRA...")
    trainer.setup_model()

    # Prepare dataset
    print("\nPreparing dataset...")
    train_dataset = trainer.prepare_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
    )

    # Train
    print("\nStarting training...")
    trainer.train(train_dataset, project_root / output_dir)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
