"""Qwen2-VL Fine-tuning for Defect Detection.

Uses LoRA to fine-tune Qwen2-VL for industrial defect detection
with bounding box output.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class BoundingBox:
    """Bounding box in YOLO format."""
    x_center: float
    y_center: float
    width: float
    height: float

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Convert to [x1, y1, x2, y2] format."""
        x1 = self.x_center - self.width / 2
        y1 = self.y_center - self.height / 2
        x2 = self.x_center + self.width / 2
        y2 = self.y_center + self.height / 2
        return (x1, y1, x2, y2)


def load_yolo_annotations(label_path: Path) -> list[tuple[int, BoundingBox]]:
    """Load YOLO format annotations."""
    if not label_path.exists():
        return []

    annotations = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                bbox = BoundingBox(
                    x_center=float(parts[1]),
                    y_center=float(parts[2]),
                    width=float(parts[3]),
                    height=float(parts[4]),
                )
                annotations.append((class_id, bbox))
    return annotations


def format_detections_as_json(
    annotations: list[tuple[int, BoundingBox]],
    class_names: list[str],
) -> str:
    """Format annotations as JSON string for training."""
    if not annotations:
        return '{"defects": []}'

    defects = []
    for class_id, bbox in annotations:
        label = class_names[class_id] if class_id < len(class_names) else "defect"
        x1, y1, x2, y2 = bbox.to_xyxy()
        defects.append(
            f'{{"label": "{label}", "bbox": [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}], "confidence": 1.0}}'
        )

    return '{"defects": [' + ", ".join(defects) + ']}'


class DefectDetectionDataset(Dataset):
    """Dataset for Qwen2-VL defect detection fine-tuning."""

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        processor: AutoProcessor,
        class_names: list[str],
        system_prompt: str,
        user_template: str,
        max_length: int = 1024,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.processor = processor
        self.class_names = class_names
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.max_length = max_length

        # Find all image files
        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.png"))
        )

        print(f"Found {len(self.image_files)} images in {images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_path = self.image_files[idx]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load annotations
        annotations = load_yolo_annotations(label_path)

        # Format as JSON response
        response_json = format_detections_as_json(annotations, self.class_names)

        # Create conversation format for Qwen2-VL
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.user_template}
                ]
            },
            {
                "role": "assistant",
                "content": response_json
            }
        ]

        # Process with Qwen2-VL processor
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        inputs = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Flatten batch dimension
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0) if "pixel_values" in inputs else None,
            "image_grid_thw": inputs.get("image_grid_thw", torch.tensor([[1, 1, 1]])).squeeze(0),
            "labels": inputs["input_ids"].squeeze(0).clone(),
        }


class QwenVLTrainer:
    """Trainer for Qwen2-VL defect detection."""

    def __init__(self, config_path: Path | str):
        self.config = self._load_config(config_path)
        self.model = None
        self.processor = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def _load_config(self, config_path: Path | str) -> dict[str, Any]:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path) as f:
            return yaml.safe_load(f)

    def setup_model(self) -> None:
        """Load model and apply LoRA."""
        model_config = self.config["model"]
        model_name = model_config["name"]

        print(f"Loading model: {model_name}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=model_config.get("trust_remote_code", True),
        )

        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=model_config.get("trust_remote_code", True),
            torch_dtype=torch.float32,  # MPS needs float32
        )

        # Apply LoRA
        lora_config = self.config["lora"]
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["dropout"],
            bias=lora_config["bias"],
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        # Move to device
        self.model.to(self.device)

    def prepare_dataset(
        self,
        images_dir: Path,
        labels_dir: Path,
    ) -> DefectDetectionDataset:
        """Prepare training dataset."""
        prompts = self.config["prompts"]
        dataset_config = self.config["dataset"]

        return DefectDetectionDataset(
            images_dir=images_dir,
            labels_dir=labels_dir,
            processor=self.processor,
            class_names=dataset_config["defect_types"],
            system_prompt=prompts["system"],
            user_template=prompts["user_template"],
            max_length=self.config["data"]["max_length"],
        )

    def train(
        self,
        train_dataset: DefectDetectionDataset,
        output_dir: Path | str,
    ) -> None:
        """Train the model."""
        training_config = self.config["training"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=training_config["num_train_epochs"],
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            learning_rate=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
            warmup_ratio=training_config["warmup_ratio"],
            lr_scheduler_type=training_config["lr_scheduler_type"],
            logging_steps=training_config["logging_steps"],
            save_steps=training_config["save_steps"],
            save_total_limit=training_config["save_total_limit"],
            fp16=False,  # MPS doesn't support fp16
            bf16=False,
            gradient_checkpointing=training_config["gradient_checkpointing"],
            max_grad_norm=training_config["max_grad_norm"],
            dataloader_num_workers=training_config["dataloader_num_workers"],
            remove_unused_columns=training_config["remove_unused_columns"],
            report_to="none",  # Disable wandb for now
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        # Train
        print(f"Starting training for {training_config['num_train_epochs']} epochs...")
        trainer.train()

        # Save final model
        print(f"Saving model to {output_dir}")
        trainer.save_model(str(output_dir / "final"))
        self.processor.save_pretrained(str(output_dir / "final"))

    def generate(
        self,
        image_path: Path | str,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate detection output for an image."""
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")

        prompts = self.config["prompts"]

        messages = [
            {
                "role": "system",
                "content": prompts["system"]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompts["user_template"]}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode
        generated_text = self.processor.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        # Extract assistant response
        if "assistant" in generated_text.lower():
            parts = generated_text.split("assistant")
            if len(parts) > 1:
                return parts[-1].strip()

        return generated_text


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Qwen2-VL for defect detection")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/qwen_vl.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing training images",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        required=True,
        help="Directory containing YOLO format labels",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/qwen_vl/real",
        help="Output directory for trained model",
    )

    args = parser.parse_args()

    # Create trainer
    trainer = QwenVLTrainer(args.config)

    # Setup model
    trainer.setup_model()

    # Prepare dataset
    train_dataset = trainer.prepare_dataset(
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
    )

    # Train
    trainer.train(train_dataset, args.output_dir)


if __name__ == "__main__":
    main()
