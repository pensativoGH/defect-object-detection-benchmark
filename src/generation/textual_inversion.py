"""Textual Inversion-based defect synthesis.

Trains concept tokens for each defect class, then uses them for generation.
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


class TextualInversionTrainer:
    """Train textual inversion embeddings for defect classes."""

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "mps",
        output_dir: Path = Path("models/textual_inversion"),
        learning_rate: float = 5e-4,
        max_train_steps: int = 2000,
        save_steps: int = 500,
        gradient_accumulation_steps: int = 4,
    ):
        self.model_id = model_id
        self.device = device
        self.output_dir = Path(output_dir)
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
        num_images: int = 10
    ) -> Path:
        """Extract and prepare training images for a specific defect class."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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

        # Select subset and crop around defects
        selected = selected[:num_images]

        for i, (img_path, label_path) in enumerate(selected):
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]

            # Find bboxes for this class
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

                        # Expand crop region with context
                        margin = max(x2 - x1, y2 - y1) // 2
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(w, x2 + margin)
                        y2 = min(h, y2 + margin)

                        # Crop and resize to 512x512
                        crop = img[y1:y2, x1:x2]
                        crop = cv2.resize(crop, (512, 512))

                        out_path = output_dir / f"defect_{i:03d}.jpg"
                        cv2.imwrite(str(out_path), crop)
                        break

        return output_dir

    def train(
        self,
        training_images_dir: Path,
        placeholder_token: str,
        initializer_token: str = "damage",
        class_name: str = "defect"
    ) -> Path:
        """Train textual inversion embedding for a defect class."""
        from accelerate import Accelerator
        from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
        from diffusers.optimization import get_scheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        from torch.utils.data import Dataset, DataLoader
        from tqdm import tqdm

        print(f"Training textual inversion for '{placeholder_token}'")
        print(f"Training images: {training_images_dir}")

        # Load models
        tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet")
        noise_scheduler = DDPMScheduler.from_pretrained(self.model_id, subfolder="scheduler")

        # Add placeholder token
        num_added_tokens = tokenizer.add_tokens(placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(f"Token {placeholder_token} already exists")

        # Resize token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialize with initializer token
        token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
        initializer_token_id = token_ids[0]
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

        # Copy initializer embedding to placeholder
        token_embeds = text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id].clone()

        # Freeze all except the new embedding
        vae.requires_grad_(False)
        unet.requires_grad_(False)
        text_encoder.text_model.encoder.requires_grad_(False)
        text_encoder.text_model.final_layer_norm.requires_grad_(False)
        text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

        # Dataset
        class TextualInversionDataset(Dataset):
            def __init__(self, images_dir, tokenizer, placeholder_token, size=512):
                self.images = list(Path(images_dir).glob("*.jpg"))
                self.tokenizer = tokenizer
                self.placeholder_token = placeholder_token
                self.size = size
                self.templates = [
                    f"a photo of a {placeholder_token}",
                    f"a rendering of a {placeholder_token}",
                    f"a close-up photo of a {placeholder_token}",
                    f"an image of a {placeholder_token}",
                    f"a photo of the {placeholder_token}",
                ]

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img = Image.open(self.images[idx]).convert("RGB")
                img = img.resize((self.size, self.size), Image.LANCZOS)
                img = np.array(img).astype(np.float32) / 255.0
                img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
                img = torch.from_numpy(img).permute(2, 0, 1)

                text = random.choice(self.templates)
                inputs = self.tokenizer(
                    text, padding="max_length", truncation=True,
                    max_length=self.tokenizer.model_max_length, return_tensors="pt"
                )
                return {"pixel_values": img, "input_ids": inputs.input_ids.squeeze()}

        dataset = TextualInversionDataset(training_images_dir, tokenizer, placeholder_token)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Move to device
        text_encoder.to(self.device)
        vae.to(self.device)
        unet.to(self.device)

        # Get embedding layer and set requires_grad
        embedding_layer = text_encoder.get_input_embeddings()
        embedding_layer.weight.requires_grad_(True)

        # Store original embeddings (to restore non-placeholder tokens)
        orig_embeds_params = embedding_layer.weight.data.clone()

        # Optimizer - optimize all embeddings but we'll mask gradients
        optimizer = torch.optim.AdamW(
            [embedding_layer.weight],
            lr=self.learning_rate
        )

        # Training loop
        text_encoder.train()
        global_step = 0
        progress_bar = tqdm(total=self.max_train_steps, desc="Training")

        while global_step < self.max_train_steps:
            for batch in dataloader:
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)

                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=self.device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                encoder_hidden_states = text_encoder(input_ids)[0]

                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                loss.backward()

                # Zero gradients for all tokens except placeholder
                grads = embedding_layer.weight.grad
                if grads is not None:
                    # Create index mask - only keep gradient for placeholder token
                    index_mask = torch.zeros(grads.shape[0], device=self.device, dtype=torch.bool)
                    index_mask[placeholder_token_id] = True
                    grads.data[~index_mask] = 0

                optimizer.step()
                optimizer.zero_grad()

                # Restore original embeddings for non-placeholder tokens
                with torch.no_grad():
                    index_no_updates = torch.ones(len(tokenizer), dtype=torch.bool)
                    index_no_updates[placeholder_token_id] = False
                    embedding_layer.weight[index_no_updates] = orig_embeds_params[index_no_updates]

                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())

                if global_step % self.save_steps == 0:
                    self._save_embedding(text_encoder, tokenizer, placeholder_token,
                                        self.output_dir / f"{class_name}_step_{global_step}")

                if global_step >= self.max_train_steps:
                    break

        progress_bar.close()

        # Save final embedding
        output_path = self.output_dir / class_name
        self._save_embedding(text_encoder, tokenizer, placeholder_token, output_path)

        print(f"Saved embedding to {output_path}")
        return output_path

    def _save_embedding(self, text_encoder, tokenizer, placeholder_token, output_path):
        """Save the learned embedding."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get the embedding
        learned_embeds = text_encoder.get_input_embeddings().weight[
            tokenizer.convert_tokens_to_ids(placeholder_token)
        ]
        learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}

        torch.save(learned_embeds_dict, output_path / "learned_embeds.bin")

        # Save token info
        with open(output_path / "token_info.txt", "w") as f:
            f.write(f"placeholder_token: {placeholder_token}\n")


class TextualInversionGenerator:
    """Generate synthetic defects using trained textual inversion embeddings."""

    CLASS_TOKENS = {
        0: "<cable-break>",
        1: "<cable-thunderbolt>"
    }

    CLASS_INITIALIZERS = {
        0: "damage",
        1: "burn"
    }

    GENERATION_PROMPTS = {
        0: "a photo of a cable with <cable-break> defect, industrial inspection, close-up",
        1: "a photo of a cable with <cable-thunderbolt> defect, industrial inspection, close-up"
    }

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        output_dir: Path,
        embeddings_dir: Path = Path("models/textual_inversion"),
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "mps",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.model_id = model_id
        self.device = device
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

        # Create output dirs
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)

        self.pipe = None
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
        """Load the model and embeddings."""
        if self.pipe is not None:
            return

        from diffusers import StableDiffusionPipeline

        print(f"Loading model: {self.model_id}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            safety_checker=None
        )
        self.pipe = self.pipe.to(self.device)

        # Load embeddings
        if class_ids is None:
            class_ids = list(self.CLASS_TOKENS.keys())

        for class_id in class_ids:
            token = self.CLASS_TOKENS[class_id]
            class_name = {0: "break", 1: "thunderbolt"}[class_id]
            embedding_path = self.embeddings_dir / class_name / "learned_embeds.bin"

            if embedding_path.exists():
                print(f"Loading embedding for class {class_id}: {token}")
                self.pipe.load_textual_inversion(str(embedding_path.parent), token=token)
            else:
                print(f"Warning: No embedding found for class {class_id} at {embedding_path}")

        print("Model and embeddings loaded")

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
        h, w = base_image.shape[:2]

        # Get prompt
        prompt = self.GENERATION_PROMPTS.get(defect_class, self.GENERATION_PROMPTS[0])

        # Generate full image
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
            height=512,
            width=512
        ).images[0]

        result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        # Find region to place on base image
        region = self._create_random_mask_region(h, w, existing_bboxes)
        if region is None:
            # Just resize and return full generated image
            result_np = cv2.resize(result_np, (w, h))
            # Defect is roughly center of image
            new_bbox = BBox(defect_class, 0.5, 0.5, 0.3, 0.3)
            return result_np, new_bbox

        x1, y1, x2, y2 = region

        # Resize generated defect to fit region
        defect_crop = cv2.resize(result_np, (x2 - x1, y2 - y1))

        # Blend into base image
        output = base_image.copy()

        # Create soft blend mask
        mask = np.ones((y2 - y1, x2 - x1), dtype=np.float32)
        blur_size = min(y2 - y1, x2 - x1) // 4
        if blur_size > 0:
            mask = cv2.GaussianBlur(mask, (blur_size * 2 + 1, blur_size * 2 + 1), 0)
        mask = mask[:, :, np.newaxis]

        # Blend
        output[y1:y2, x1:x2] = (defect_crop * mask + output[y1:y2, x1:x2] * (1 - mask)).astype(np.uint8)

        # Create bbox
        new_bbox = BBox.from_xyxy(defect_class, x1, y1, x2, y2, w, h)

        return output, new_bbox

    def generate(self, num_samples: int, seed: int = 42) -> int:
        """Generate synthetic samples."""
        self.setup()

        random.seed(seed)
        np.random.seed(seed)

        generated = 0
        class_ids = list(self.CLASS_TOKENS.keys())

        for i in range(num_samples):
            # Pick random sample as base
            sample = random.choice(self.samples)

            # Load image
            image = cv2.imread(str(sample["image_path"]))
            if image is None:
                continue

            # Pick random defect class
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
            out_name = f"synth_ti_{i:05d}"
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
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()


def train_textual_inversion(
    data_dir: Path,
    output_dir: Path = Path("models/textual_inversion"),
    max_train_steps: int = 2000,
    learning_rate: float = 5e-4
) -> dict:
    """Train textual inversion for all defect classes."""
    data_dir = Path(data_dir)

    trainer = TextualInversionTrainer(
        output_dir=output_dir,
        max_train_steps=max_train_steps,
        learning_rate=learning_rate
    )

    results = {}
    class_names = {0: "break", 1: "thunderbolt"}
    tokens = TextualInversionGenerator.CLASS_TOKENS
    initializers = TextualInversionGenerator.CLASS_INITIALIZERS

    for class_id, class_name in class_names.items():
        print(f"\n{'='*50}")
        print(f"Training for class {class_id}: {class_name}")
        print(f"{'='*50}")

        # Prepare training images
        train_images_dir = output_dir / f"train_images_{class_name}"
        trainer.prepare_training_images(
            data_dir / "images",
            data_dir / "labels",
            class_id,
            train_images_dir,
            num_images=15
        )

        # Train
        embedding_path = trainer.train(
            train_images_dir,
            placeholder_token=tokens[class_id],
            initializer_token=initializers[class_id],
            class_name=class_name
        )

        results[class_name] = str(embedding_path)

        # Cleanup training images
        shutil.rmtree(train_images_dir, ignore_errors=True)

    return results


def generate_textual_inversion_dataset(
    data_dir: Path,
    output_dir: Path,
    embeddings_dir: Path = Path("models/textual_inversion"),
    num_samples: int = 100,
    seed: int = 42
) -> dict:
    """Generate synthetic dataset using textual inversion."""
    generator = TextualInversionGenerator(
        images_dir=data_dir / "images",
        labels_dir=data_dir / "labels",
        output_dir=output_dir,
        embeddings_dir=embeddings_dir
    )

    try:
        generated = generator.generate(num_samples, seed)
    finally:
        generator.cleanup()

    return {
        "method": "textual_inversion",
        "generated": generated,
        "output_dir": str(output_dir)
    }


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/processed/Cable/train")

    if mode == "train":
        output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("models/textual_inversion")
        result = train_textual_inversion(data_dir, output_dir)
        print(f"Training complete: {result}")

    elif mode == "generate":
        output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/synthetic/textual_inversion")
        embeddings_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("models/textual_inversion")
        num_samples = int(sys.argv[5]) if len(sys.argv) > 5 else 100

        result = generate_textual_inversion_dataset(data_dir, output_dir, embeddings_dir, num_samples)
        print(f"Generated {result['generated']} samples")
