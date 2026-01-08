"""YOLO training wrapper with experiment tracking."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import yaml
import wandb
from ultralytics import YOLO


class YOLOTrainer:
    """Wrapper for YOLO training with experiment management."""

    def __init__(
        self,
        experiment_name: str,
        data_yaml: Path,
        config_path: Optional[Path] = None,
        output_dir: Path = Path("experiments"),
        use_wandb: bool = True,
        wandb_project: str = "synthetic-defect-comparison"
    ):
        self.experiment_name = experiment_name
        self.data_yaml = Path(data_yaml)
        self.output_dir = Path(output_dir) / experiment_name
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        # Load config
        self.config = self._load_config(config_path)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load training configuration."""
        default_config = {
            'model': 'yolov8m.pt',
            'epochs': 100,
            'imgsz': 512,
            'batch': 8,
            'device': 'mps',
            'patience': 20,
            'save_period': 10,
            'workers': 4,
            'augment': True,
        }

        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)

        return default_config

    def train(
        self,
        resume: bool = False,
        **override_config
    ) -> Dict[str, Any]:
        """
        Train YOLO model.

        Args:
            resume: Resume from last checkpoint
            **override_config: Override any config values

        Returns:
            Training results dict
        """
        # Merge configs
        train_config = {**self.config, **override_config}

        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                name=self.experiment_name,
                config=train_config
            )

        # Save experiment metadata
        metadata = {
            'name': self.experiment_name,
            'data_yaml': str(self.data_yaml),
            'config': train_config,
            'timestamp': datetime.now().isoformat(),
        }
        with open(self.output_dir / 'experiment.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Initialize model
        model = YOLO(train_config['model'])

        # Train
        results = model.train(
            data=str(self.data_yaml),
            epochs=train_config['epochs'],
            imgsz=train_config['imgsz'],
            batch=train_config['batch'],
            device=train_config['device'],
            patience=train_config['patience'],
            save_period=train_config['save_period'],
            workers=train_config['workers'],
            project=str(self.output_dir.parent),
            name=self.experiment_name,
            exist_ok=True,
            resume=resume,
            augment=train_config.get('augment', True),
            hsv_h=train_config.get('hsv_h', 0.015),
            hsv_s=train_config.get('hsv_s', 0.7),
            hsv_v=train_config.get('hsv_v', 0.4),
            degrees=train_config.get('degrees', 0),
            translate=train_config.get('translate', 0.1),
            scale=train_config.get('scale', 0.5),
            fliplr=train_config.get('fliplr', 0.5),
            mosaic=train_config.get('mosaic', 1.0),
            plots=True,
        )

        # Log final metrics
        if self.use_wandb:
            wandb.log({
                'final/mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
                'final/mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
                'final/precision': results.results_dict.get('metrics/precision(B)', 0),
                'final/recall': results.results_dict.get('metrics/recall(B)', 0),
            })
            wandb.finish()

        # Save results
        results_dict = {
            'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
            'precision': results.results_dict.get('metrics/precision(B)', 0),
            'recall': results.results_dict.get('metrics/recall(B)', 0),
            'best_model': str(self.output_dir / 'weights' / 'best.pt'),
        }

        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)

        return results_dict

    def evaluate(
        self,
        model_path: Optional[Path] = None,
        data_yaml: Optional[Path] = None,
        split: str = 'test'
    ) -> Dict[str, Any]:
        """
        Evaluate trained model.

        Args:
            model_path: Path to model weights (default: best.pt from training)
            data_yaml: Data config (default: training data)
            split: Dataset split to evaluate on

        Returns:
            Evaluation metrics dict
        """
        if model_path is None:
            model_path = self.output_dir / 'weights' / 'best.pt'

        if data_yaml is None:
            data_yaml = self.data_yaml

        model = YOLO(str(model_path))

        results = model.val(
            data=str(data_yaml),
            split=split,
            device=self.config.get('device', 'mps'),
        )

        eval_results = {
            'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
            'precision': results.results_dict.get('metrics/precision(B)', 0),
            'recall': results.results_dict.get('metrics/recall(B)', 0),
        }

        # Per-class metrics if available
        if hasattr(results, 'ap_class_index'):
            eval_results['per_class_ap'] = {
                results.names[i]: float(results.maps[i])
                for i in results.ap_class_index
            }

        return eval_results


def train_baseline(
    data_yaml: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
    use_wandb: bool = True
) -> Dict[str, Any]:
    """Train baseline YOLO without synthetic data."""
    trainer = YOLOTrainer(
        experiment_name="baseline_real_only",
        data_yaml=data_yaml,
        config_path=config_path,
        output_dir=output_dir,
        use_wandb=use_wandb
    )
    return trainer.train(augment=False)


def train_with_augmentation(
    data_yaml: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
    use_wandb: bool = True
) -> Dict[str, Any]:
    """Train YOLO with traditional augmentation."""
    trainer = YOLOTrainer(
        experiment_name="traditional_augmentation",
        data_yaml=data_yaml,
        config_path=config_path,
        output_dir=output_dir,
        use_wandb=use_wandb
    )
    return trainer.train(augment=True)
