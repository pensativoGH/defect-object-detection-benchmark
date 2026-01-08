"""Copy-Paste augmentation for defect synthesis."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import random
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


class CopyPasteGenerator:
    """Generate synthetic defect images using copy-paste augmentation."""

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        output_dir: Path,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        rotation_range: Tuple[int, int] = (-15, 15),
        blend_mode: str = "poisson"  # "poisson", "alpha", "direct"
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.blend_mode = blend_mode

        # Create output dirs
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)

        # Load all samples
        self.samples = self._load_samples()

    def _load_samples(self) -> List[dict]:
        """Load all image-label pairs."""
        samples = []
        for img_path in sorted(self.images_dir.glob("*.jpg")):
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                bboxes = self._load_bboxes(label_path)
                if bboxes:
                    samples.append({
                        "image_path": img_path,
                        "label_path": label_path,
                        "bboxes": bboxes
                    })
        return samples

    def _load_bboxes(self, label_path: Path) -> List[BBox]:
        """Load bboxes from YOLO format label file."""
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

    def _extract_defect(self, image: np.ndarray, bbox: BBox) -> Tuple[np.ndarray, np.ndarray]:
        """Extract defect region and create mask."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox.to_xyxy(w, h)

        # Add small padding
        pad = 5
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        defect_region = image[y1:y2, x1:x2].copy()

        # Create elliptical mask for smoother blending
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        center = ((x2 - x1) // 2, (y2 - y1) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        return defect_region, mask

    def _get_clean_region(self, image: np.ndarray, bboxes: List[BBox],
                          defect_size: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find a clean region to paste defect."""
        h, w = image.shape[:2]
        dw, dh = defect_size

        # Create mask of occupied regions
        occupied = np.zeros((h, w), dtype=np.uint8)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.to_xyxy(w, h)
            # Add margin around existing defects
            margin = 20
            x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
            x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
            occupied[y1:y2, x1:x2] = 255

        # Try random positions
        for _ in range(100):
            x = random.randint(0, max(0, w - dw))
            y = random.randint(0, max(0, h - dh))

            # Check if region is clean
            if occupied[y:y+dh, x:x+dw].sum() == 0:
                return x, y

        return None

    def _blend_images(self, base: np.ndarray, defect: np.ndarray,
                      mask: np.ndarray, x: int, y: int) -> np.ndarray:
        """Blend defect onto base image."""
        result = base.copy()
        dh, dw = defect.shape[:2]

        # Ensure we don't go out of bounds
        bh, bw = base.shape[:2]
        if x + dw > bw or y + dh > bh:
            return result

        if self.blend_mode == "poisson":
            try:
                center = (x + dw // 2, y + dh // 2)
                result = cv2.seamlessClone(defect, result, mask, center, cv2.NORMAL_CLONE)
            except:
                # Fallback to alpha blend
                self.blend_mode = "alpha"
                return self._blend_images(base, defect, mask, x, y)

        elif self.blend_mode == "alpha":
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            roi = result[y:y+dh, x:x+dw]
            blended = (defect * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
            result[y:y+dh, x:x+dw] = blended

        else:  # direct
            mask_bool = mask > 127
            result[y:y+dh, x:x+dw][mask_bool] = defect[mask_bool]

        return result

    def _transform_defect(self, defect: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random transformations to defect."""
        # Random scale
        scale = random.uniform(*self.scale_range)
        new_w = int(defect.shape[1] * scale)
        new_h = int(defect.shape[0] * scale)
        if new_w > 0 and new_h > 0:
            defect = cv2.resize(defect, (new_w, new_h))
            mask = cv2.resize(mask, (new_w, new_h))

        # Random rotation
        angle = random.randint(*self.rotation_range)
        if angle != 0:
            h, w = defect.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            defect = cv2.warpAffine(defect, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h))

        # Random flip
        if random.random() > 0.5:
            defect = cv2.flip(defect, 1)
            mask = cv2.flip(mask, 1)

        return defect, mask

    def generate(self, num_samples: int, seed: int = 42) -> int:
        """Generate synthetic samples."""
        random.seed(seed)
        np.random.seed(seed)

        generated = 0

        for i in range(num_samples):
            # Pick random source and target
            src_sample = random.choice(self.samples)
            tgt_sample = random.choice(self.samples)

            # Load images
            src_img = cv2.imread(str(src_sample["image_path"]))
            tgt_img = cv2.imread(str(tgt_sample["image_path"]))

            if src_img is None or tgt_img is None:
                continue

            # Pick random defect from source
            src_bbox = random.choice(src_sample["bboxes"])

            # Extract and transform defect
            defect, mask = self._extract_defect(src_img, src_bbox)
            defect, mask = self._transform_defect(defect, mask)

            # Find clean region in target
            paste_pos = self._get_clean_region(tgt_img, tgt_sample["bboxes"],
                                               (defect.shape[1], defect.shape[0]))

            if paste_pos is None:
                continue

            # Blend
            result = self._blend_images(tgt_img, defect, mask, *paste_pos)

            # Create new bbox
            px, py = paste_pos
            dh, dw = defect.shape[:2]
            th, tw = tgt_img.shape[:2]
            new_bbox = BBox.from_xyxy(src_bbox.class_id, px, py, px + dw, py + dh, tw, th)

            # Combine with existing bboxes
            all_bboxes = tgt_sample["bboxes"] + [new_bbox]

            # Save
            out_name = f"synth_copypaste_{i:05d}"
            cv2.imwrite(str(self.output_dir / "images" / f"{out_name}.jpg"), result)

            with open(self.output_dir / "labels" / f"{out_name}.txt", "w") as f:
                for bbox in all_bboxes:
                    f.write(bbox.to_yolo() + "\n")

            generated += 1

        return generated


def generate_copypaste_dataset(
    data_dir: Path,
    output_dir: Path,
    num_samples: int = 100,
    seed: int = 42
) -> dict:
    """Generate copy-paste synthetic dataset."""
    generator = CopyPasteGenerator(
        images_dir=data_dir / "images",
        labels_dir=data_dir / "labels",
        output_dir=output_dir,
        blend_mode="poisson"
    )

    generated = generator.generate(num_samples, seed)

    return {
        "method": "copy_paste",
        "generated": generated,
        "output_dir": str(output_dir)
    }


if __name__ == "__main__":
    import sys

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/Cable/train")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/synthetic/copypaste")
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    result = generate_copypaste_dataset(data_dir, output_dir, num_samples)
    print(f"Generated {result['generated']} samples")
