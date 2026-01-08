"""VLM Evaluation module for defect detection.

Evaluates Vision Language Models on industrial defect detection tasks
using the OpenRouter API.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml
from PIL import Image
from tqdm import tqdm


@dataclass
class BoundingBox:
    """Bounding box for detection result."""
    x1: float
    y1: float
    x2: float
    y2: float

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_yolo(self) -> tuple[float, float, float, float]:
        """Convert to YOLO format (x_center, y_center, width, height)."""
        x_center = (self.x1 + self.x2) / 2
        y_center = (self.y1 + self.y2) / 2
        width = self.x2 - self.x1
        height = self.y2 - self.y1
        return (x_center, y_center, width, height)

    @classmethod
    def from_list(cls, coords: list[float]) -> "BoundingBox":
        """Create from list of coordinates."""
        if len(coords) != 4:
            raise ValueError(f"Expected 4 coordinates, got {len(coords)}")
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

    @classmethod
    def from_yolo(cls, x_center: float, y_center: float, width: float, height: float) -> "BoundingBox":
        """Create from YOLO format."""
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)

    def is_valid(self) -> bool:
        """Check if bounding box is valid."""
        return (
            0 <= self.x1 <= 1 and
            0 <= self.y1 <= 1 and
            0 <= self.x2 <= 1 and
            0 <= self.y2 <= 1 and
            self.x1 < self.x2 and
            self.y1 < self.y2
        )


@dataclass
class Detection:
    """Single detection result."""
    bbox: BoundingBox
    label: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.to_list(),
            "label": self.label,
            "confidence": self.confidence,
        }


@dataclass
class DetectionResult:
    """Result from a detection task."""
    image_path: str
    defects: list[Detection] = field(default_factory=list)
    raw_response: str = ""
    parse_success: bool = True
    latency_ms: float = 0.0

    @property
    def has_defect(self) -> bool:
        return len(self.defects) > 0

    @property
    def num_defects(self) -> int:
        return len(self.defects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "has_defect": self.has_defect,
            "num_defects": self.num_defects,
            "defects": [d.to_dict() for d in self.defects],
            "parse_success": self.parse_success,
            "latency_ms": self.latency_ms,
        }


@dataclass
class CostTracker:
    """Track API costs across requests."""
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    request_count: int = 0

    def add(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        self.total_cost += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.request_count += 1

    def summary(self) -> dict[str, Any]:
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_requests": self.request_count,
        }


class ResponseParser:
    """Parse VLM responses into structured detection results."""

    JSON_PATTERNS = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'(\{[\s\S]*\})',
    ]

    def __init__(self, class_names: list[str]):
        self.class_names = [c.lower() for c in class_names]

    def parse(self, response: str, image_path: str) -> DetectionResult:
        """Parse a model response to detection result."""
        data = self._extract_json(response)

        if data is not None:
            try:
                defects_data = data.get("defects", [])
                defects = []

                for d in defects_data:
                    bbox_raw = d.get("bbox", [0, 0, 0, 0])
                    bbox_normalized = self._normalize_bbox(bbox_raw)

                    try:
                        bbox = BoundingBox.from_list(bbox_normalized)
                        if not bbox.is_valid():
                            continue
                    except ValueError:
                        continue

                    label = d.get("label", "defect").lower()
                    # Map to valid class name
                    if label not in self.class_names:
                        label = self._find_closest_class(label)

                    confidence = float(d.get("confidence", 0.5))
                    confidence = max(0.0, min(1.0, confidence))

                    defects.append(Detection(
                        bbox=bbox,
                        label=label,
                        confidence=confidence,
                    ))

                return DetectionResult(
                    image_path=image_path,
                    defects=defects,
                    raw_response=response,
                    parse_success=True,
                )
            except (KeyError, TypeError, ValueError):
                pass

        # Fallback
        return DetectionResult(
            image_path=image_path,
            defects=[],
            raw_response=response,
            parse_success=False,
        )

    def _extract_json(self, response: str) -> dict[str, Any] | None:
        """Try to extract JSON from a response string."""
        for pattern in self.JSON_PATTERNS:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                try:
                    cleaned = match.strip()
                    data = json.loads(cleaned)
                    if isinstance(data, dict):
                        return data
                except (json.JSONDecodeError, ValueError):
                    continue
        return None

    def _normalize_bbox(self, bbox: list[float]) -> list[float]:
        """Ensure bounding box coordinates are normalized (0-1)."""
        if any(v > 1.0 for v in bbox):
            max_val = max(bbox)
            if max_val > 100:
                return bbox  # Can't normalize without image size
            elif max_val > 1:
                return [v / 100.0 for v in bbox]
        return bbox

    def _find_closest_class(self, label: str) -> str:
        """Find closest matching class name."""
        label_lower = label.lower()
        for class_name in self.class_names:
            if class_name in label_lower or label_lower in class_name:
                return class_name
        return self.class_names[0] if self.class_names else "defect"


class VLMEvaluator:
    """Evaluator for Vision Language Models on defect detection."""

    def __init__(
        self,
        config_path: Path | str,
        api_key: str | None = None,
    ):
        self.config = self._load_config(config_path)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.base_url = self.config["api"]["base_url"]
        self.timeout = self.config["api"]["timeout_seconds"]
        self.cost_tracker = CostTracker()

        # Setup parser
        class_names = self.config["dataset"]["defect_types"]
        self.parser = ResponseParser(class_names)

    def _load_config(self, config_path: Path | str) -> dict[str, Any]:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def encode_image(image_path: Path | str) -> str:
        """Encode an image file to base64."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def get_image_media_type(image_path: Path | str) -> str:
        """Get the media type for an image file."""
        suffix = Path(image_path).suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
        }
        return media_types.get(suffix, "image/jpeg")

    def _get_model_config(self, model_name: str) -> dict[str, Any]:
        """Get model configuration."""
        models = self.config.get("models", {})
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
        return models[model_name]

    def _calculate_cost(
        self,
        model_config: dict[str, Any],
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for a request."""
        input_cost = (input_tokens / 1000) * model_config.get("cost_per_1k_input", 0)
        output_cost = (output_tokens / 1000) * model_config.get("cost_per_1k_output", 0)
        return input_cost + output_cost

    async def _make_request(
        self,
        model_config: dict[str, Any],
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Make a request to the OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/synthetic-defect-comparison",
            "X-Title": "Synthetic Defect Comparison",
        }

        payload = {
            "model": model_config["openrouter_id"],
            "messages": messages,
            "max_tokens": max_tokens or model_config.get("max_tokens", 4096),
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(3):
                try:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code == 429:
                        # Rate limited, wait and retry
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    else:
                        print(f"API Error {response.status_code}: {response.text[:500]}")
                        response.raise_for_status()
                except httpx.TimeoutException:
                    if attempt < 2:
                        await asyncio.sleep(2)
                        continue
                    raise

        raise Exception("Failed after 3 retries")

    async def evaluate_image(
        self,
        model_name: str,
        image_path: Path | str,
        prompt_type: str = "zero_shot",
        few_shot_examples: list[dict] | None = None,
    ) -> DetectionResult:
        """Evaluate a single image.

        Args:
            model_name: Name of the model to use
            image_path: Path to the image file
            prompt_type: "zero_shot" or "few_shot"
            few_shot_examples: List of example dicts with 'image_path' and 'annotation'

        Returns:
            DetectionResult with detected defects
        """
        model_config = self._get_model_config(model_name)
        image_path = Path(image_path)

        # Build prompt
        prompts = self.config.get("prompts", {})
        system_prompt = prompts.get("system", "You are an expert quality inspector.")

        if prompt_type == "zero_shot":
            user_prompt = prompts.get("zero_shot", "Analyze this image for defects.")
        else:
            # Few-shot with examples
            user_prompt = prompts.get("few_shot_prefix", "Here are examples:\n")
            if few_shot_examples:
                for i, ex in enumerate(few_shot_examples):
                    user_prompt += f"\nExample {i+1}: {ex.get('annotation', 'defect detected')}\n"
            user_prompt += prompts.get("few_shot_suffix", "\nNow analyze the new image.")

        # Encode image
        image_base64 = self.encode_image(image_path)
        media_type = self.get_image_media_type(image_path)

        # Build messages for Gemini (no system role)
        if "gemini" in model_name.lower():
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_base64}"
                            }
                        },
                        {"type": "text", "text": combined_prompt}
                    ]
                }
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_base64}"
                            }
                        },
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]

        # Make request
        start_time = time.time()
        raw_response = await self._make_request(model_config, messages)
        latency_ms = (time.time() - start_time) * 1000

        # Extract response
        content = raw_response["choices"][0]["message"]["content"]
        usage = raw_response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # Calculate and track cost
        cost = self._calculate_cost(model_config, input_tokens, output_tokens)
        self.cost_tracker.add(input_tokens, output_tokens, cost)

        # Parse response
        result = self.parser.parse(content, str(image_path))
        result.latency_ms = latency_ms

        return result

    async def evaluate_batch(
        self,
        model_name: str,
        image_paths: list[Path | str],
        prompt_type: str = "zero_shot",
        few_shot_examples: list[dict] | None = None,
        concurrency: int = 5,
        show_progress: bool = True,
    ) -> list[DetectionResult]:
        """Evaluate multiple images with concurrency control."""
        semaphore = asyncio.Semaphore(concurrency)

        async def limited_evaluate(image_path: Path | str) -> DetectionResult:
            async with semaphore:
                try:
                    return await self.evaluate_image(
                        model_name, image_path, prompt_type, few_shot_examples
                    )
                except Exception as e:
                    return DetectionResult(
                        image_path=str(image_path),
                        defects=[],
                        raw_response=str(e),
                        parse_success=False,
                    )

        if show_progress:
            results = []
            for image_path in tqdm(image_paths, desc=f"Evaluating {model_name}"):
                result = await limited_evaluate(image_path)
                results.append(result)
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
            return results
        else:
            tasks = [limited_evaluate(img) for img in image_paths]
            return await asyncio.gather(*tasks)

    def evaluate_batch_sync(
        self,
        model_name: str,
        image_paths: list[Path | str],
        prompt_type: str = "zero_shot",
        few_shot_examples: list[dict] | None = None,
        concurrency: int = 5,
    ) -> list[DetectionResult]:
        """Synchronous wrapper for batch evaluation."""
        return asyncio.run(
            self.evaluate_batch(
                model_name, image_paths, prompt_type, few_shot_examples, concurrency
            )
        )

    def get_cost_summary(self) -> dict[str, Any]:
        """Get summary of API costs."""
        return self.cost_tracker.summary()

    def reset_cost_tracker(self) -> None:
        """Reset the cost tracker."""
        self.cost_tracker = CostTracker()


def load_yolo_annotations(label_path: Path | str) -> list[tuple[int, float, float, float, float]]:
    """Load YOLO format annotations from a label file.

    Returns list of (class_id, x_center, y_center, width, height)
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return []

    annotations = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                annotations.append((class_id, x_center, y_center, width, height))

    return annotations


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute IoU between two bounding boxes."""
    # Calculate intersection
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Calculate union
    area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def evaluate_detections(
    predictions: list[DetectionResult],
    labels_dir: Path | str,
    class_names: list[str],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate detection predictions against ground truth.

    Args:
        predictions: List of DetectionResult objects
        labels_dir: Directory containing YOLO format label files
        class_names: List of class names
        iou_threshold: IoU threshold for matching

    Returns:
        Dictionary with evaluation metrics
    """
    labels_dir = Path(labels_dir)

    # Collect all predictions and ground truths
    all_tp = 0
    all_fp = 0
    all_fn = 0

    per_class_tp = {c: 0 for c in class_names}
    per_class_fp = {c: 0 for c in class_names}
    per_class_fn = {c: 0 for c in class_names}

    parse_failures = 0

    for pred in predictions:
        if not pred.parse_success:
            parse_failures += 1

        # Load ground truth
        image_name = Path(pred.image_path).stem
        label_path = labels_dir / f"{image_name}.txt"
        gt_annotations = load_yolo_annotations(label_path)

        # Convert GT to bboxes
        gt_boxes = []
        for class_id, x_c, y_c, w, h in gt_annotations:
            bbox = BoundingBox.from_yolo(x_c, y_c, w, h)
            label = class_names[class_id] if class_id < len(class_names) else "unknown"
            gt_boxes.append((bbox, label))

        # Match predictions to ground truth
        matched_gt = set()

        for detection in pred.defects:
            best_iou = 0
            best_gt_idx = -1

            for i, (gt_bbox, gt_label) in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                if detection.label != gt_label:
                    continue

                iou = compute_iou(detection.bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # True positive
                all_tp += 1
                per_class_tp[detection.label] += 1
                matched_gt.add(best_gt_idx)
            else:
                # False positive
                all_fp += 1
                per_class_fp[detection.label] += 1

        # Count false negatives (unmatched ground truths)
        for i, (gt_bbox, gt_label) in enumerate(gt_boxes):
            if i not in matched_gt:
                all_fn += 1
                if gt_label in per_class_fn:
                    per_class_fn[gt_label] += 1

    # Calculate metrics
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    per_class_metrics = {}
    for c in class_names:
        tp = per_class_tp[c]
        fp = per_class_fp[c]
        fn = per_class_fn[c]

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0

        per_class_metrics[c] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_tp": all_tp,
        "total_fp": all_fp,
        "total_fn": all_fn,
        "parse_failures": parse_failures,
        "per_class": per_class_metrics,
    }
