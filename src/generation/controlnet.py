"""ControlNet-based defect synthesis with edge conditioning."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
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


class ControlNetGenerator:
    """Generate synthetic defects using ControlNet with Canny edge conditioning."""

    DEFECT_PROMPTS = {
        0: "industrial cable with visible break damage, broken wire, electrical defect, damaged insulation",
        1: "cable with burn mark from electrical discharge, thunderbolt damage, scorch mark, heat damage"
    }

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        output_dir: Path,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/sd-controlnet-canny",
        device: str = "mps",
        canny_low: int = 100,
        canny_high: int = 200,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        num_inference_steps: int = 30,
        defect_region_size: Tuple[float, float] = (0.05, 0.15)
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.device = device
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.guidance_scale = guidance_scale
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.num_inference_steps = num_inference_steps
        self.defect_region_size = defect_region_size

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
        """Load the ControlNet pipeline."""
        if self.pipe is not None:
            return

        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

        print(f"Loading ControlNet: {self.controlnet_id}")
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id,
            torch_dtype=torch.float32
        )

        print(f"Loading SD model: {self.model_id}")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=controlnet,
            torch_dtype=torch.float32,
            safety_checker=None
        )
        self.pipe = self.pipe.to(self.device)
        print("Models loaded")

    def _get_canny_edges(self, image: np.ndarray) -> np.ndarray:
        """Extract Canny edges from image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        return edges

    def _select_defect_region(
        self, h: int, w: int, existing_bboxes: List[BBox]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Select a region to focus defect generation, avoiding existing bboxes."""
        # Region size
        region_w = int(w * random.uniform(*self.defect_region_size))
        region_h = int(h * random.uniform(*self.defect_region_size))

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
            x = random.randint(0, max(0, w - region_w))
            y = random.randint(0, max(0, h - region_h))
            if occupied[y:y+region_h, x:x+region_w].sum() == 0:
                return x, y, x + region_w, y + region_h

        return None

    def _modify_edges_for_defect(
        self, edges: np.ndarray, region: Tuple[int, int, int, int], defect_class: int
    ) -> np.ndarray:
        """Modify edge map in the defect region to guide generation."""
        modified = edges.copy()
        x1, y1, x2, y2 = region

        if defect_class == 0:  # break - add irregular break pattern
            # Add some random lines to simulate break edges
            for _ in range(random.randint(2, 5)):
                pt1 = (random.randint(x1, x2), random.randint(y1, y2))
                pt2 = (random.randint(x1, x2), random.randint(y1, y2))
                cv2.line(modified, pt1, pt2, 255, random.randint(1, 2))
        else:  # thunderbolt - add starburst pattern
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            for _ in range(random.randint(3, 6)):
                angle = random.uniform(0, 2 * np.pi)
                length = random.randint(10, min(x2-x1, y2-y1) // 2)
                end_pt = (
                    int(center[0] + length * np.cos(angle)),
                    int(center[1] + length * np.sin(angle))
                )
                cv2.line(modified, center, end_pt, 255, 1)

        return modified

    def generate_single(
        self, image: np.ndarray, existing_bboxes: List[BBox],
        defect_class: int, seed: int
    ) -> Optional[Tuple[np.ndarray, BBox]]:
        """Generate a single defect on the image using ControlNet."""
        h, w = image.shape[:2]

        # Select region for defect
        region = self._select_defect_region(h, w, existing_bboxes)
        if region is None:
            return None

        # Get edge map
        edges = self._get_canny_edges(image)

        # Modify edges in defect region
        modified_edges = self._modify_edges_for_defect(edges, region, defect_class)

        # Convert to PIL (3-channel for ControlNet)
        edges_3ch = cv2.cvtColor(modified_edges, cv2.COLOR_GRAY2RGB)
        canny_image = Image.fromarray(edges_3ch)

        # Resize to 512x512 for SD
        orig_size = (w, h)
        canny_image = canny_image.resize((512, 512), Image.LANCZOS)

        # Get prompt
        prompt = self.DEFECT_PROMPTS.get(defect_class, self.DEFECT_PROMPTS[0])

        # Generate
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            image=canny_image,
            guidance_scale=self.guidance_scale,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator
        ).images[0]

        # Resize back
        result = result.resize(orig_size, Image.LANCZOS)
        result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        # Blend only the defect region back into original
        x1, y1, x2, y2 = region

        # Create smooth blend mask
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = mask[:, :, np.newaxis]

        # Blend generated result with original
        blended = (result_np * mask + image * (1 - mask)).astype(np.uint8)

        # Create bbox
        new_bbox = BBox.from_xyxy(defect_class, x1, y1, x2, y2, w, h)

        return blended, new_bbox

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
            out_name = f"synth_controlnet_{i:05d}"
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


def generate_controlnet_dataset(
    data_dir: Path,
    output_dir: Path,
    num_samples: int = 50,
    seed: int = 42
) -> dict:
    """Generate ControlNet synthetic dataset."""
    generator = ControlNetGenerator(
        images_dir=data_dir / "images",
        labels_dir=data_dir / "labels",
        output_dir=output_dir
    )

    try:
        generated = generator.generate(num_samples, seed)
    finally:
        generator.cleanup()

    return {
        "method": "controlnet",
        "generated": generated,
        "output_dir": str(output_dir)
    }


if __name__ == "__main__":
    import sys

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/Cable/train")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/synthetic/controlnet")
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    result = generate_controlnet_dataset(data_dir, output_dir, num_samples)
    print(f"Generated {result['generated']} samples")
