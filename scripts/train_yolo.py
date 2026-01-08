#!/usr/bin/env python3
"""YOLO training script with experiment management."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import json
import yaml
from src.training.yolo_trainer import YOLOTrainer


@click.command()
@click.option('--experiment-name', '-n', required=True, help='Experiment name')
@click.option('--data-yaml', '-d', required=True, type=click.Path(exists=True), help='Path to data.yaml')
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to training config YAML')
@click.option('--output-dir', '-o', default='experiments', help='Output directory')
@click.option('--epochs', '-e', type=int, help='Override number of epochs')
@click.option('--batch', '-b', type=int, help='Override batch size')
@click.option('--imgsz', '-i', type=int, help='Override image size')
@click.option('--device', type=str, help='Override device (mps, cuda, cpu)')
@click.option('--augment/--no-augment', default=True, help='Enable/disable augmentation')
@click.option('--wandb/--no-wandb', default=True, help='Enable/disable W&B logging')
@click.option('--resume', is_flag=True, help='Resume from last checkpoint')
def train(
    experiment_name: str,
    data_yaml: str,
    config: str,
    output_dir: str,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    augment: bool,
    wandb: bool,
    resume: bool
):
    """Train YOLO model for defect detection."""
    click.echo(f"Starting experiment: {experiment_name}")

    # Build override config
    override = {}
    if epochs:
        override['epochs'] = epochs
    if batch:
        override['batch'] = batch
    if imgsz:
        override['imgsz'] = imgsz
    if device:
        override['device'] = device
    override['augment'] = augment

    # Create trainer
    trainer = YOLOTrainer(
        experiment_name=experiment_name,
        data_yaml=Path(data_yaml),
        config_path=Path(config) if config else None,
        output_dir=Path(output_dir),
        use_wandb=wandb
    )

    # Train
    results = trainer.train(resume=resume, **override)

    # Print results
    click.echo("\n" + "=" * 50)
    click.echo("Training Complete!")
    click.echo("=" * 50)
    click.echo(f"mAP@0.5:     {results['mAP50']:.4f}")
    click.echo(f"mAP@0.5:0.95: {results['mAP50-95']:.4f}")
    click.echo(f"Precision:   {results['precision']:.4f}")
    click.echo(f"Recall:      {results['recall']:.4f}")
    click.echo(f"\nBest model saved to: {results['best_model']}")


@click.command()
@click.option('--model', '-m', required=True, type=click.Path(exists=True), help='Path to model weights')
@click.option('--data-yaml', '-d', required=True, type=click.Path(exists=True), help='Path to data.yaml')
@click.option('--split', '-s', default='test', help='Dataset split (train/val/test)')
@click.option('--device', type=str, default='mps', help='Device (mps, cuda, cpu)')
@click.option('--output', '-o', type=click.Path(), help='Output JSON file for results')
def evaluate(
    model: str,
    data_yaml: str,
    split: str,
    device: str,
    output: str
):
    """Evaluate trained YOLO model."""
    from ultralytics import YOLO

    click.echo(f"Evaluating model: {model}")
    click.echo(f"Data: {data_yaml}")
    click.echo(f"Split: {split}")

    yolo = YOLO(model)
    results = yolo.val(data=data_yaml, split=split, device=device)

    eval_results = {
        'model': model,
        'data': data_yaml,
        'split': split,
        'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
        'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
        'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
        'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
    }

    # Print results
    click.echo("\n" + "=" * 50)
    click.echo("Evaluation Results")
    click.echo("=" * 50)
    click.echo(f"mAP@0.5:     {eval_results['mAP50']:.4f}")
    click.echo(f"mAP@0.5:0.95: {eval_results['mAP50-95']:.4f}")
    click.echo(f"Precision:   {eval_results['precision']:.4f}")
    click.echo(f"Recall:      {eval_results['recall']:.4f}")

    # Save if output specified
    if output:
        with open(output, 'w') as f:
            json.dump(eval_results, f, indent=2)
        click.echo(f"\nResults saved to: {output}")

    return eval_results


@click.group()
def cli():
    """YOLO Training CLI for Synthetic Defect Comparison."""
    pass


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == '__main__':
    cli()
