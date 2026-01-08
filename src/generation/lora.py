"""LoRA-based defect synthesis.

Fine-tunes Stable Diffusion using LoRA for each defect class, then uses them for generation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random
import torch
from PIL import Image
from dataclasses import dataclass
import shutil
import json


@dataclass
class BBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_xyxy(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        x1 = int((self.x_center - self.width / 2) * img_w)
        y1 = int((self.y_center - self.height / 2) * img_h)
        x2 = int((self.x_center + self.width / 2) * img_w)
        y2 = int((self.y_center + self.height / 2) * img_h)
        return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

    @classmethod
    def from_xyxy(cls, class_id: int, x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int):
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        return cls(class_id, x_center, y_center, width, height)

    def to_yolo(self) -> str:
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


class LoRATrainer:
    """Train LoRA weights for Stable Diffusion on defect images."""

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "mps",
        output_dir: Path = Path("models/lora"),
        rank: int = 4,
        alpha: int = 4,
        learning_rate: float = 1e-4,
        max_train_steps: int = 1000,
        save_steps: int = 250,
        gradient_accumulation_steps: int = 4,
    ):
        self.model_id = model_id
        self.device = device
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_train_steps = max_train_steps
        self.save_steps = save_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_training_images(
        self,
        images_dir: Path,
        labels_dir: Path,
        class_id: int,
        output_dir: Path,
        num_images: int = 20
    ) -> Tuple[Path, Path]:
        """Extract and prepare training images for a specific defect class."""
        output_dir = Path(output_dir)
        images_out = output_dir / "images"
        images_out.mkdir(parents=True, exist_ok=True)

        # Find images containing this class
        selected = []
        for label_path in sorted(labels_dir.glob("*.txt")):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) == class_id:
                        img_path = images_dir / f"{label_path.stem}.jpg"
                        if img_path.exists():
                            selected.append((img_path, label_path))
                        break

        if not selected:
            raise ValueError(f"No images found for class {class_id}")

        # Select subset
        selected = selected[:num_images]

        # Create metadata file for captions
        metadata = []

        for i, (img_path, label_path) in enumerate(selected):
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]

            # Find bboxes for this class and crop around defect
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) == class_id:
                        bbox = BBox(
                            class_id=int(parts[0]),
                            x_center=float(parts[1]),
                            y_center=float(parts[2]),
                            width=float(parts[3]),
                            height=float(parts[4])
                        )
                        x1, y1, x2, y2 = bbox.to_xyxy(w, h)

                        # Expand crop region
                        margin = max(x2 - x1, y2 - y1)
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(w, x2 + margin)
                        y2 = min(h, y2 + margin)

                        # Crop and resize
                        crop = img[y1:y2, x1:x2]
                        crop = cv2.resize(crop, (512, 512))

                        out_name = f"defect_{i:03d}.jpg"
                        cv2.imwrite(str(images_out / out_name), crop)

                        # Add to metadata
                        metadata.append({
                            "file_name": out_name,
                            "text": self._get_caption(class_id)
                        })
                        break

        # Save metadata
        metadata_path = output_dir / "metadata.jsonl"
        with open(metadata_path, "w") as f:
            for item in metadata:
                f.write(json.dumps(item) + "\n")

        return images_out, metadata_path

    def _get_caption(self, class_id: int) -> str:
        """Get training caption for class."""
        captions = {
            0: "a photo of a cable with break defect, damaged wire, industrial inspection",
            1: "a photo of a cable with thunderbolt damage, burn mark, electrical discharge"
        }
        return captions.get(class_id, captions[0])

    def train(
        self,
        training_data_dir: Path,
        class_name: str = "defect"
    ) -> Path:
        """Train LoRA weights for a defect class."""
        from diffusers import (
            AutoencoderKL,
            DDPMScheduler,
            StableDiffusionPipeline,
            UNet2DConditionModel
        )
        from transformers import CLIPTextModel, CLIPTokenizer
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import Dataset, DataLoader
        from tqdm import tqdm

        print(f"\nTraining LoRA for class: {class_name}")
        print(f"Training data: {training_data_dir}")
        print(f"LoRA rank: {self.rank}, alpha: {self.alpha}")

        # Load models
        tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet")
        noise_scheduler = DDPMScheduler.from_pretrained(self.model_id, subfolder="scheduler")

        # Freeze VAE and text encoder
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        # Apply LoRA to UNet
        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.0,
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()

        # Dataset
        class LoRADataset(Dataset):
            def __init__(self, images_dir, metadata_path, tokenizer, size=512):
                self.images_dir = Path(images_dir)
                self.tokenizer = tokenizer
                self.size = size

                # Load metadata
                self.samples = []
                with open(metadata_path) as f:
                    for line in f:
                        item = json.loads(line)
                        self.samples.append(item)

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                sample = self.samples[idx]
                img_path = self.images_dir / sample["file_name"]

                img = Image.open(img_path).convert("RGB")
                img = img.resize((self.size, self.size), Image.LANCZOS)
                img = np.array(img).astype(np.float32) / 255.0
                img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
                img = torch.from_numpy(img).permute(2, 0, 1)

                text = sample["text"]
                inputs = self.tokenizer(
                    text, padding="max_length", truncation=True,
                    max_length=self.tokenizer.model_max_length, return_tensors="pt"
                )
                return {"pixel_values": img, "input_ids": inputs.input_ids.squeeze()}

        images_dir = training_data_dir / "images"
        metadata_path = training_data_dir / "metadata.jsonl"
        dataset = LoRADataset(images_dir, metadata_path, tokenizer)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Move to device
        text_encoder.to(self.device)
        vae.to(self.device)
        unet.to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(unet.parameters(), lr=self.learning_rate)

        # Training loop
        unet.train()
        global_step = 0
        progress_bar = tqdm(total=self.max_train_steps, desc="Training LoRA")

        while global_step < self.max_train_steps:
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)

                # Encode images
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=self.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())

                if global_step % self.save_steps == 0:
                    self._save_lora(unet, self.output_dir / f"{class_name}_step_{global_step}")

                if global_step >= self.max_train_steps:
                    break

        progress_bar.close()

        # Save final weights
        output_path = self.output_dir / class_name
        self._save_lora(unet, output_path)

        print(f"Saved LoRA weights to {output_path}")
        return output_path

    def _save_lora(self, model, output_path):
        """Save LoRA weights."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)


class LoRAGenerator:
    """Generate synthetic defects using trained LoRA models."""

    CLASS_NAMES = {
        0: "break",
        1: "thunderbolt"
    }

    GENERATION_PROMPTS = {
        0: "a photo of a cable with break defect, damaged wire, industrial inspection, close-up",
        1: "a photo of a cable with thunderbolt damage, burn mark, electrical discharge, industrial inspection"
    }

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        output_dir: Path,
        lora_dir: Path = Path("models/lora"),
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "mps",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        lora_scale: float = 0.8
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        self.lora_dir = Path(lora_dir)
        self.model_id = model_id
        self.device = device
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.lora_scale = lora_scale

        # Create output dirs
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)

        self.pipes = {}  # One pipe per class
        self.samples = self._load_samples()

    def _load_samples(self) -> List[dict]:
        """Load all image-label pairs."""
        samples = []
        for img_path in sorted(self.images_dir.glob("*.jpg")):
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            bboxes = []
            if label_path.exists():
                bboxes = self._load_bboxes(label_path)
            samples.append({
                "image_path": img_path,
                "label_path": label_path,
                "bboxes": bboxes
            })
        return samples

    def _load_bboxes(self, label_path: Path) -> List[BBox]:
        """Load bboxes from YOLO format."""
        bboxes = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    bboxes.append(BBox(
                        class_id=int(parts[0]),
                        x_center=float(parts[1]),
                        y_center=float(parts[2]),
                        width=float(parts[3]),
                        height=float(parts[4])
                    ))
        return bboxes

    def setup(self, class_ids: List[int] = None):
        """Load models with LoRA weights."""
        if class_ids is None:
            class_ids = list(self.CLASS_NAMES.keys())

        from diffusers import StableDiffusionPipeline

        for class_id in class_ids:
            if class_id in self.pipes:
                continue

            class_name = self.CLASS_NAMES[class_id]
            lora_path = self.lora_dir / class_name

            if not lora_path.exists():
                print(f"Warning: No LoRA weights found for class {class_id} at {lora_path}")
                continue

            print(f"Loading model with LoRA for class {class_id}: {class_name}")

            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                safety_checker=None
            )

            # Load LoRA weights
            pipe.load_lora_weights(str(lora_path))

            pipe = pipe.to(self.device)
            self.pipes[class_id] = pipe

        print(f"Loaded {len(self.pipes)} LoRA models")

    def _create_random_mask_region(self, h: int, w: int, existing_bboxes: List[BBox],
                                   mask_size_range: Tuple[float, float] = (0.1, 0.2)) -> Optional[Tuple[int, int, int, int]]:
        """Find a random clean region for defect placement."""
        mask_w = int(w * random.uniform(*mask_size_range))
        mask_h = int(h * random.uniform(*mask_size_range))

        # Create occupied map
        occupied = np.zeros((h, w), dtype=np.uint8)
        for bbox in existing_bboxes:
            x1, y1, x2, y2 = bbox.to_xyxy(w, h)
            margin = 20
            x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
            x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
            occupied[y1:y2, x1:x2] = 255

        # Find clean position
        for _ in range(100):
            x = random.randint(0, max(0, w - mask_w))
            y = random.randint(0, max(0, h - mask_h))
            if occupied[y:y+mask_h, x:x+mask_w].sum() == 0:
                return (x, y, x + mask_w, y + mask_h)

        return None

    def generate_single(self, base_image: np.ndarray, existing_bboxes: List[BBox],
                        defect_class: int, seed: int) -> Optional[Tuple[np.ndarray, BBox]]:
        """Generate a single defect image."""
        if defect_class not in self.pipes:
            print(f"No pipe for class {defect_class}")
            return None

        h, w = base_image.shape[:2]
        pipe = self.pipes[defect_class]

        # Get prompt
        prompt = self.GENERATION_PROMPTS.get(defect_class, self.GENERATION_PROMPTS[0])

        # Generate
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = pipe(
            prompt=prompt,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            height=512,
            width=512,
            cross_attention_kwargs={"scale": self.lora_scale}
        ).images[0]

        result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        # Find region to blend into base image
        region = self._create_random_mask_region(h, w, existing_bboxes)
        if region is None:
            # Return full generated image resized
            result_np = cv2.resize(result_np, (w, h))
            new_bbox = BBox(defect_class, 0.5, 0.5, 0.3, 0.3)
            return result_np, new_bbox

        x1, y1, x2, y2 = region

        # Resize generated defect to region
        defect_crop = cv2.resize(result_np, (x2 - x1, y2 - y1))

        # Blend into base image
        output = base_image.copy()

        # Soft blend mask
        mask = np.ones((y2 - y1, x2 - x1), dtype=np.float32)
        blur_size = min(y2 - y1, x2 - x1) // 4
        if blur_size > 0:
            mask = cv2.GaussianBlur(mask, (blur_size * 2 + 1, blur_size * 2 + 1), 0)
        mask = mask[:, :, np.newaxis]

        output[y1:y2, x1:x2] = (defect_crop * mask + output[y1:y2, x1:x2] * (1 - mask)).astype(np.uint8)

        new_bbox = BBox.from_xyxy(defect_class, x1, y1, x2, y2, w, h)
        return output, new_bbox

    def generate(self, num_samples: int, seed: int = 42) -> int:
        """Generate synthetic samples."""
        self.setup()

        if not self.pipes:
            print("No LoRA models loaded!")
            return 0

        random.seed(seed)
        np.random.seed(seed)

        generated = 0
        class_ids = list(self.pipes.keys())

        for i in range(num_samples):
            # Pick random base image
            sample = random.choice(self.samples)

            # Load image
            image = cv2.imread(str(sample["image_path"]))
            if image is None:
                continue

            # Pick random defect class (from available)
            defect_class = random.choice(class_ids)

            # Generate
            result = self.generate_single(
                image,
                sample["bboxes"],
                defect_class,
                seed=seed + i
            )

            if result is None:
                continue

            result_img, new_bbox = result

            # Combine bboxes
            all_bboxes = sample["bboxes"] + [new_bbox]

            # Save
            out_name = f"synth_lora_{i:05d}"
            cv2.imwrite(str(self.output_dir / "images" / f"{out_name}.jpg"), result_img)

            with open(self.output_dir / "labels" / f"{out_name}.txt", "w") as f:
                for bbox in all_bboxes:
                    f.write(bbox.to_yolo() + "\n")

            generated += 1
            if generated % 10 == 0:
                print(f"Generated {generated}/{num_samples}")

        return generated

    def cleanup(self):
        """Release model resources."""
        for pipe in self.pipes.values():
            del pipe
        self.pipes = {}
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def train_lora(
    data_dir: Path,
    output_dir: Path = Path("models/lora"),
    rank: int = 4,
    max_train_steps: int = 1000,
    learning_rate: float = 1e-4
) -> dict:
    """Train LoRA for all defect classes."""
    data_dir = Path(data_dir)

    trainer = LoRATrainer(
        output_dir=output_dir,
        rank=rank,
        alpha=rank,
        max_train_steps=max_train_steps,
        learning_rate=learning_rate
    )

    results = {}
    class_names = {0: "break", 1: "thunderbolt"}

    for class_id, class_name in class_names.items():
        print(f"\n{'='*50}")
        print(f"Training LoRA for class {class_id}: {class_name}")
        print(f"{'='*50}")

        # Prepare training data
        train_data_dir = output_dir / f"train_data_{class_name}"
        trainer.prepare_training_images(
            data_dir / "images",
            data_dir / "labels",
            class_id,
            train_data_dir,
            num_images=20
        )

        # Train
        lora_path = trainer.train(train_data_dir, class_name)
        results[class_name] = str(lora_path)

        # Cleanup training data
        shutil.rmtree(train_data_dir, ignore_errors=True)

    return results


def generate_lora_dataset(
    data_dir: Path,
    output_dir: Path,
    lora_dir: Path = Path("models/lora"),
    num_samples: int = 100,
    seed: int = 42
) -> dict:
    """Generate synthetic dataset using LoRA models."""
    generator = LoRAGenerator(
        images_dir=data_dir / "images",
        labels_dir=data_dir / "labels",
        output_dir=output_dir,
        lora_dir=lora_dir
    )

    try:
        generated = generator.generate(num_samples, seed)
    finally:
        generator.cleanup()

    return {
        "method": "lora",
        "generated": generated,
        "output_dir": str(output_dir)
    }


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/processed/Cable/train")

    if mode == "train":
        output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("models/lora")
        result = train_lora(data_dir, output_dir)
        print(f"Training complete: {result}")

    elif mode == "generate":
        output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/synthetic/lora")
        lora_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("models/lora")
        num_samples = int(sys.argv[5]) if len(sys.argv) > 5 else 100

        result = generate_lora_dataset(data_dir, output_dir, lora_dir, num_samples)
        print(f"Generated {result['generated']} samples")
