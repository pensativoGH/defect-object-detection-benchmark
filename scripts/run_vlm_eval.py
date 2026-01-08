#!/usr/bin/env python3
"""Run VLM evaluation on defect detection dataset.

Usage:
    python scripts/run_vlm_eval.py --model gemini-2.0-flash --mode zero_shot
    python scripts/run_vlm_eval.py --model gemini-2.0-flash --mode few_shot --examples real
    python scripts/run_vlm_eval.py --model gemini-2.0-flash --mode few_shot --examples synthetic
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.vlm_eval import (
    VLMEvaluator,
    evaluate_detections,
    load_yolo_annotations,
    BoundingBox,
)


def get_few_shot_examples(
    source: str,
    images_dir: Path,
    labels_dir: Path,
    class_names: list[str],
    num_examples: int = 3,
) -> list[dict]:
    """Get few-shot examples from real or synthetic data.

    Args:
        source: "real" or "synthetic"
        images_dir: Directory containing images
        labels_dir: Directory containing labels
        class_names: List of class names
        num_examples: Number of examples per class

    Returns:
        List of example dicts with image_path and annotation description
    """
    examples = []

    # Find images with annotations
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        annotations = load_yolo_annotations(label_path)

        if annotations:
            # Create annotation description
            defect_descriptions = []
            for class_id, x_c, y_c, w, h in annotations:
                label = class_names[class_id] if class_id < len(class_names) else "defect"
                bbox = BoundingBox.from_yolo(x_c, y_c, w, h)
                defect_descriptions.append(
                    f'{{"label": "{label}", "bbox": [{bbox.x1:.3f}, {bbox.y1:.3f}, {bbox.x2:.3f}, {bbox.y2:.3f}], "confidence": 0.95}}'
                )

            annotation = f'{{"defects": [{", ".join(defect_descriptions)}]}}'

            examples.append({
                "image_path": str(image_path),
                "annotation": annotation,
            })

            if len(examples) >= num_examples * len(class_names):
                break

    return examples[:num_examples * len(class_names)]


def main():
    parser = argparse.ArgumentParser(description="Run VLM evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Model name (gemini-2.0-flash, gemini-2.5-pro, gpt-4o)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "few_shot"],
        help="Evaluation mode",
    )
    parser.add_argument(
        "--examples",
        type=str,
        default="real",
        choices=["real", "synthetic_ti", "synthetic_lora", "synthetic_copypaste"],
        help="Source of few-shot examples (only for few_shot mode)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of few-shot examples per class",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to evaluate (for testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/vlm_eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Number of concurrent API requests",
    )

    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "evaluation" / "vlm_models.yaml"

    # Validation data
    val_images_dir = project_root / "data" / "processed" / "Cable" / "val" / "images"
    val_labels_dir = project_root / "data" / "processed" / "Cable" / "val" / "labels"

    # Training data for few-shot examples
    train_images_dir = project_root / "data" / "processed" / "Cable" / "train" / "images"
    train_labels_dir = project_root / "data" / "processed" / "Cable" / "train" / "labels"

    # Synthetic data directories
    synthetic_dirs = {
        "synthetic_ti": project_root / "data" / "synthetic" / "textual_inversion",
        "synthetic_lora": project_root / "data" / "synthetic" / "lora",
        "synthetic_copypaste": project_root / "data" / "synthetic" / "copy_paste",
    }

    # Check API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set it with: export OPENROUTER_API_KEY='your-key'")
        sys.exit(1)

    # Class names
    class_names = ["break", "thunderbolt"]

    # Initialize evaluator
    print(f"Initializing VLM evaluator with model: {args.model}")
    evaluator = VLMEvaluator(config_path)

    # Get validation images
    image_paths = sorted(val_images_dir.glob("*.jpg")) + sorted(val_images_dir.glob("*.png"))
    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print(f"Found {len(image_paths)} validation images")

    # Get few-shot examples if needed
    few_shot_examples = None
    if args.mode == "few_shot":
        if args.examples == "real":
            examples_images_dir = train_images_dir
            examples_labels_dir = train_labels_dir
        else:
            synth_dir = synthetic_dirs.get(args.examples)
            if synth_dir and synth_dir.exists():
                examples_images_dir = synth_dir / "images"
                examples_labels_dir = synth_dir / "labels"
            else:
                print(f"Error: Synthetic data directory not found: {synth_dir}")
                sys.exit(1)

        few_shot_examples = get_few_shot_examples(
            args.examples,
            examples_images_dir,
            examples_labels_dir,
            class_names,
            args.num_examples,
        )
        print(f"Using {len(few_shot_examples)} few-shot examples from {args.examples}")

    # Run evaluation
    print(f"\nRunning {args.mode} evaluation...")
    prompt_type = "zero_shot" if args.mode == "zero_shot" else "few_shot"

    results = evaluator.evaluate_batch_sync(
        model_name=args.model,
        image_paths=image_paths,
        prompt_type=prompt_type,
        few_shot_examples=few_shot_examples,
        concurrency=args.concurrency,
    )

    # Compute metrics
    print("\nComputing metrics...")
    metrics = evaluate_detections(
        predictions=results,
        labels_dir=val_labels_dir,
        class_names=class_names,
        iou_threshold=0.5,
    )

    # Get cost summary
    cost_summary = evaluator.get_cost_summary()

    # Print results
    print("\n" + "=" * 60)
    print(f"VLM Evaluation Results - {args.model} ({args.mode})")
    print("=" * 60)
    print(f"\nOverall Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  TP: {metrics['total_tp']}, FP: {metrics['total_fp']}, FN: {metrics['total_fn']}")
    print(f"  Parse failures: {metrics['parse_failures']}")

    print(f"\nPer-Class Metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1:        {class_metrics['f1']:.4f}")

    print(f"\nAPI Cost Summary:")
    print(f"  Total requests: {cost_summary['total_requests']}")
    print(f"  Total cost: ${cost_summary['total_cost_usd']:.4f}")
    print(f"  Input tokens: {cost_summary['total_input_tokens']}")
    print(f"  Output tokens: {cost_summary['total_output_tokens']}")

    # Save results
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = f"{args.model}_{args.mode}"
    if args.mode == "few_shot":
        experiment_name += f"_{args.examples}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"{experiment_name}_{timestamp}.json"

    output_data = {
        "experiment": {
            "model": args.model,
            "mode": args.mode,
            "examples_source": args.examples if args.mode == "few_shot" else None,
            "num_images": len(image_paths),
            "timestamp": timestamp,
        },
        "metrics": metrics,
        "cost_summary": cost_summary,
        "predictions": [r.to_dict() for r in results],
    }

    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
