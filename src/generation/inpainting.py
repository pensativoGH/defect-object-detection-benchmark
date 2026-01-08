"""SD Inpainting-based defect synthesis."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random
import torch
from PIL import Image
from dataclasses import dataclass


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


class InpaintingGenerator:
    """Generate synthetic defects using SD Inpainting."""

    DEFECT_PROMPTS = {
        0: "damaged cable with visible break, broken wire, electrical damage, industrial defect",
        1: "cable with thunderbolt damage mark, burn mark, electrical discharge damage, scorch mark"
    }

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        output_dir: Path,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        device: str = "mps",
        mask_size_range: Tuple[float, float] = (0.05, 0.15),
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        self.model_id = model_id
        self.device = device
        self.mask_size_range = mask_size_range
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

    def setup(self):
        """Load the inpainting model."""
        if self.pipe is not None:
            return

        from diffusers import StableDiffusionInpaintPipeline

        print(f"Loading inpainting model: {self.model_id}")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,  # MPS works better with float32
            safety_checker=None
        )
        self.pipe = self.pipe.to(self.device)
        print("Model loaded")

    def _create_random_mask(self, h: int, w: int, existing_bboxes: List[BBox]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Create a random mask in a clean region."""
        # Mask size
        mask_w = int(w * random.uniform(*self.mask_size_range))
        mask_h = int(h * random.uniform(*self.mask_size_range))

        # Create occupied map
        occupied = np.zeros((h, w), dtype=np.uint8)
        for bbox in existing_bboxes:
            x1, y1, x2, y2 = bbox.to_xyxy(w, h)
            margin = 30
            x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
            x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
            occupied[y1:y2, x1:x2] = 255

        # Find clean position
        for _ in range(100):
            x = random.randint(0, max(0, w - mask_w))
            y = random.randint(0, max(0, h - mask_h))
            if occupied[y:y+mask_h, x:x+mask_w].sum() == 0:
                # Create elliptical mask
                mask = np.zeros((h, w), dtype=np.uint8)
                center = (x + mask_w // 2, y + mask_h // 2)
                axes = (mask_w // 2, mask_h // 2)
                cv2.ellipse(mask, center, axes, random.randint(0, 180), 0, 360, 255, -1)
                return mask, (x, y, x + mask_w, y + mask_h)

        return None, None

    def generate_single(self, image: np.ndarray, existing_bboxes: List[BBox],
                        defect_class: int, seed: int) -> Optional[Tuple[np.ndarray, BBox]]:
        """Generate a single defect on the image."""
        h, w = image.shape[:2]

        # Create mask
        mask, bbox_coords = self._create_random_mask(h, w, existing_bboxes)
        if mask is None:
            return None

        # Convert to PIL
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)

        # Resize to 512x512 for SD
        orig_size = image_pil.size
        image_pil = image_pil.resize((512, 512), Image.LANCZOS)
        mask_pil = mask_pil.resize((512, 512), Image.NEAREST)

        # Get prompt
        prompt = self.DEFECT_PROMPTS.get(defect_class, self.DEFECT_PROMPTS[0])

        # Generate
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            image=image_pil,
            mask_image=mask_pil,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator
        ).images[0]

        # Resize back
        result = result.resize(orig_size, Image.LANCZOS)
        result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        # Create bbox from mask region
        x1, y1, x2, y2 = bbox_coords
        new_bbox = BBox.from_xyxy(defect_class, x1, y1, x2, y2, w, h)

        return result_np, new_bbox

    def generate(self, num_samples: int, seed: int = 42) -> int:
        """Generate synthetic samples."""
        self.setup()

        random.seed(seed)
        np.random.seed(seed)

        generated = 0
        class_ids = list(self.DEFECT_PROMPTS.keys())

        for i in range(num_samples):
            # Pick random sample
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
            out_name = f"synth_inpaint_{i:05d}"
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


def generate_inpainting_dataset(
    data_dir: Path,
    output_dir: Path,
    num_samples: int = 50,
    seed: int = 42
) -> dict:
    """Generate inpainting synthetic dataset."""
    generator = InpaintingGenerator(
        images_dir=data_dir / "images",
        labels_dir=data_dir / "labels",
        output_dir=output_dir
    )

    try:
        generated = generator.generate(num_samples, seed)
    finally:
        generator.cleanup()

    return {
        "method": "inpainting",
        "generated": generated,
        "output_dir": str(output_dir)
    }


if __name__ == "__main__":
    import sys

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/Cable/train")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/synthetic/inpainting")
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    result = generate_inpainting_dataset(data_dir, output_dir, num_samples)
    print(f"Generated {result['generated']} samples")
