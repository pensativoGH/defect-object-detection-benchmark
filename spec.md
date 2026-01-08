# Specification: Synthetic Data for Industrial Defect Detection

**Version**: 1.0
**Date**: 2026-01-06
**Status**: Draft

---

# Project Overview

## Context
- **Data**: Mixed defect types (surface, structural, texture anomalies)
- **Examples**: 20-50 per defect category (moderate few-shot)
- **Defect Classes**: 10-20 classes
- **Class Imbalance**: Moderate (2-10x difference between classes)
- **Annotations**: Bounding boxes available
- **Image Resolution**: Low (< 512px)
- **Defect Size**: Small (5-15% of image area)
- **Good Samples**: None - use non-bbox regions as clean areas
- **Surface Type**: Mixed/Variable
- **Compute**: MacBook MPS (16GB unified memory)
- **Focus**: Diffusion-based generation + VLM comparison

## Detection Models to Compare
| Model Type | Specific Models | Training |
|------------|-----------------|----------|
| Traditional | YOLOv11+ | Full fine-tuning |
| VLM (Local) | Qwen-VL | LoRA fine-tuning |
| VLM (API) | Gemini | Zero-shot + few-shot prompting |

## VLM Output Format
```
<think>
[reasoning about defect detection]
</think>
<final>
[bounding box coordinates]
</final>
```

## Experiment Tracking
- **Tool**: Weights & Biases
- **Execution**: Python scripts (CLI-based)
- **Timeline**: 1 day with Claude Code assistance

## Base Image Strategy (No Good Samples Available)

Since all images contain defects, use this approach for synthetic generation:

```
1. Parse existing images with bbox annotations
2. Identify "clean regions" = image area OUTSIDE all bboxes
3. For Copy-Paste: Extract clean patches as backgrounds
4. For Inpainting: Mask regions adjacent to but outside bboxes
5. For ControlNet: Use edge maps from clean regions
```

**Clean Region Extraction**:
```python
def get_clean_regions(image, bboxes, min_size=64):
    """Extract patches from areas outside all bounding boxes."""
    mask = np.ones(image.shape[:2], dtype=bool)
    for bbox in bboxes:
        x, y, w, h = bbox_to_xyxy(bbox)
        mask[y:y+h, x:x+w] = False
    # Find contiguous clean regions > min_size
    return extract_patches(image, mask, min_size)
```

---

## MPS Constraints & Considerations

⚠️ **Apple MPS has limitations compared to CUDA**:
- Slower training (2-5x compared to equivalent NVIDIA GPU)
- Some operations not supported (may fall back to CPU)
- Memory typically 16-32GB unified (shared with system)
- Some libraries have incomplete MPS support

**Recommended approach for MPS**:
1. **Inference-heavy methods preferred** (SD Inpainting, ControlNet)
2. **Lightweight training** (Textual Inversion, small LoRA)
3. **Consider cloud for heavy training** (DreamBooth, full LoRA)
4. Use `torch.backends.mps.is_available()` to verify support

**MPS-Compatible Tools**:
- ✅ `diffusers` (good MPS support)
- ✅ Ultralytics YOLOv8 (excellent MPS support)
- ⚠️ `kohya-ss` (CUDA-focused, may need workarounds)
- ✅ `transformers` (good MPS support)

---

## Approaches to Compare (Diffusion-Focused)

### 1. Traditional Augmentation (Baseline)
**Description**: Standard image augmentations without generative models.

**Techniques**:
- Geometric: rotation, flipping, scaling, elastic deformation
- Photometric: brightness, contrast, color jitter, noise injection
- Advanced: CutOut, MixUp, CutMix, GridMask

**Pros**: Fast, no training needed, deterministic
**Cons**: Limited diversity, doesn't create new defect patterns
**Compute**: Minimal (CPU-based)

---

### 2. Defect Copy-Paste Augmentation
**Description**: Extract defect regions from real images and paste onto good/normal samples.

**Techniques**:
- Manual or auto-segmented defect masks
- Poisson blending for seamless compositing
- Random placement, scale, and rotation of defects
- Multiple defects per image

**Pros**: Preserves real defect characteristics, fast, no training
**Cons**: Requires defect segmentation, limited to existing defect appearances
**Compute**: Minimal

**Tools**: Albumentations, custom scripts

---

### 3. Few-Shot GAN (FastGAN / Lightweight GAN)
**Description**: GANs designed specifically for limited data scenarios.

**Techniques**:
- FastGAN: Skip-layer excitation, self-supervised discriminator
- Lightweight GAN: Augmentation-driven training
- Data augmentation in discriminator (DiffAugment, ADA)

**Pros**: Designed for 100-1000 images, single GPU friendly
**Cons**: May struggle with diverse defect types, mode collapse risk
**Compute**: 1-2 days training on single GPU

**Implementations**:
- `lucidrains/lightweight-gan`
- `odegeasslern/FastGAN-pytorch`

---

### 4. Stable Diffusion Fine-tuning
**Description**: Fine-tune diffusion models on industrial defect data.

**Techniques**:
| Method | Training Images | VRAM | Training Time |
|--------|-----------------|------|---------------|
| Textual Inversion | 3-5 | 8GB | 1-2 hours |
| LoRA | 10-50 | 12GB | 2-4 hours |
| DreamBooth | 5-30 | 16GB+ | 30 min - 2 hours |
| Full fine-tune | 1000+ | 24GB+ | Days |

**Pros**: High quality, controllable via prompts, leverages pretrained knowledge
**Cons**: Industrial domain gap, may need careful prompt engineering
**Compute**: Feasible with LoRA/DreamBooth on 24GB GPU

**Best fit for your case**: LoRA or DreamBooth per defect class

---

### 5. ControlNet + Stable Diffusion
**Description**: Generate defects with structural guidance (edges, depth, segmentation).

**Workflow**:
1. Extract edge/canny maps from good samples
2. Use ControlNet to maintain structure
3. Guide generation with defect prompts
4. Optionally train custom ControlNet on defect masks

**Pros**: Structure-preserving, controllable defect placement
**Cons**: Requires paired data for custom ControlNet training
**Compute**: Inference is feasible, custom training needs more resources

---

### 6. Inpainting-based Defect Synthesis
**Description**: Use inpainting models to "paint in" defects on good samples.

**Workflow**:
1. Start with defect-free images
2. Create random masks where defects should appear
3. Use inpainting model (SD Inpainting, LaMa) with defect prompts
4. Generate various defect types in masked regions

**Pros**: Natural blending, control over defect location
**Cons**: Quality depends on prompt engineering, may not match real defect distribution
**Compute**: Inference only, single GPU friendly

---

### 7. Domain Randomization / Procedural Generation
**Description**: Programmatically generate synthetic defects.

**Techniques**:
- Procedural scratch/crack generation (Bezier curves, fractals)
- Texture synthesis for surface anomalies
- Physics-based deformation simulation
- Perlin noise for stains/discoloration

**Pros**: Infinite diversity, fully controllable, no training
**Cons**: May lack realism, requires domain expertise to design
**Compute**: Minimal (CPU-based)

---

### 8. Hybrid Approaches
**Description**: Combine multiple methods.

**Examples**:
- Copy-paste + diffusion refinement
- Procedural generation + GAN-based style transfer
- ControlNet structure + defect texture transfer

---

## Recommended Comparison Framework (MPS-Optimized)

### Phase 1: Baselines (No Generative Training)
1. **No augmentation** - establish baseline YOLO performance
2. **Traditional augmentation** - Albumentations (geometric + photometric)
3. **Bounding-box Copy-Paste** - crop defects via bbox, paste on good samples

### Phase 2: Inference-Only Methods (MPS Friendly ✅)
4. **SD Inpainting** - mask regions on good images, prompt for defects
5. **ControlNet Canny/Edge** - maintain product structure, generate defects
6. **ControlNet + Inpainting** - hybrid for precise defect placement

### Phase 3: Lightweight Training (MPS Feasible ⚠️)
7. **Textual Inversion** - learn defect concept tokens (~2-4 hours on MPS)
8. **Small LoRA** - rank 4-8, per defect class (~4-8 hours on MPS)

### Phase 4: Cloud/Heavy Training (Optional)
9. **DreamBooth** - if cloud GPU available
10. **Full LoRA (rank 32+)** - if cloud GPU available

### Priority Order for MPS:
**High Priority**: Methods 1-6 (run locally, fast iteration)
**Medium Priority**: Methods 7-8 (local but slower)
**Low Priority**: Methods 9-10 (require cloud resources)

---

## YOLO Integration Strategy

### Synthetic Data Format
Generated images must include YOLO-format annotations:
```
<class_id> <x_center> <y_center> <width> <height>
```

### Annotation Generation Approaches

| Method | Annotation Strategy |
|--------|---------------------|
| Copy-Paste | Track paste location → automatic bbox |
| Inpainting | Use inpaint mask → convert to bbox |
| ControlNet | Use conditioning mask → bbox |
| Full image generation | Requires post-hoc detection or manual |

### Training Strategy
1. **Baseline**: Train YOLO on real data only
2. **Augmented**: Train on real + synthetic (various ratios)
3. **Pre-train + Fine-tune**: Pre-train on synthetic, fine-tune on real
4. **Curriculum**: Start with easy synthetic, progress to hard real

---

## Evaluation Metrics

### Generation Quality
- FID (Fréchet Inception Distance) per defect class
- LPIPS (perceptual similarity to real defects)
- Visual inspection / expert review
- Defect-specific: edge sharpness, texture realism

### YOLO Detection Performance
- **mAP@0.5** and **mAP@0.5:0.95** on held-out real test set
- **Per-class AP** (critical for rare defects)
- **Precision/Recall curves**
- **Confusion matrix** between defect classes

### Ablation Metrics
- Performance vs synthetic data ratio
- Performance vs number of synthetic samples
- Quality filtering impact (reject low-FID generations)

### Practical Metrics
- Generation time per image
- Fine-tuning time per method
- VRAM usage

---

## Experimental Design

```
For each generation method:
  1. Generate N synthetic images per defect class (N = 100, 500, 1000)
  2. Create augmented training sets: Real + Synthetic
  3. Fine-tune existing detection model
  4. Evaluate on held-out real test set
  5. Compare: Real-only vs Real+Synthetic
```

### Ablation Studies
- Varying synthetic-to-real ratio (1:1, 2:1, 5:1)
- Quality filtering (FID-based rejection)
- Class-balanced vs proportional generation

---

## Implementation Details per Method

### Method 1: Textual Inversion
```
Tool: diffusers / AUTOMATIC1111
Input: 10-20 images per defect class
Output: Learned token embedding (<scratch>, <crack>, etc.)
Training: ~1-2 hours on single GPU
Generation: "A photo of industrial part with <scratch> defect"
```

### Method 2: LoRA Fine-tuning
```
Tool: kohya-ss/sd-scripts or diffusers
Input: 20-50 images per defect class
Output: LoRA weights (~10-100MB per class)
Training: ~2-4 hours on single GPU
Recommended: SDXL + LoRA for higher quality
```

### Method 3: DreamBooth
```
Tool: diffusers / ShivamShrirao implementation
Input: 5-30 images per defect class
Output: Full model checkpoint or LoRA
Training: ~30 min - 2 hours
Note: Risk of overfitting with <50 images
```

### Method 4: SD Inpainting
```
Tool: diffusers InpaintingPipeline
Input: Good product images + random masks
Process:
  1. Generate random mask in plausible defect locations
  2. Inpaint with defect prompt
  3. Convert mask → bbox annotation
No training required (inference only)
```

### Method 5: ControlNet
```
Tool: diffusers ControlNetModel
Conditioning: Canny edges, segmentation, or depth
Process:
  1. Extract edge map from good product image
  2. Generate with defect prompt + edge conditioning
  3. Maintains product structure, varies defect
```

---

## Recommended Tech Stack (MPS-Compatible)

| Component | Recommended Tool | MPS Support |
|-----------|------------------|-------------|
| Diffusion | `diffusers` (HuggingFace) | ✅ Good |
| LoRA training | `peft` + `diffusers` | ✅ Good |
| ControlNet | `diffusers` + pretrained ControlNet | ✅ Good |
| Base model | SD 1.5 (lighter) or SDXL | ✅ Both work |
| Detection | Ultralytics YOLOv8 | ✅ Excellent |
| Augmentation | Albumentations | ✅ CPU-based |
| Metrics | `torchmetrics`, `clean-fid` | ✅ Works |
| Experiment tracking | Weights & Biases | ✅ Works |
| Image processing | OpenCV, PIL | ✅ Works |

**Python Version**: 3.10+ recommended for best MPS support
**PyTorch Version**: 2.0+ for stable MPS backend

---

## Project Location

**Path**: `/Users/pramodsharma/code/synthetic-defect-comparison/`

```
synthetic-defect-comparison/
├── README.md
├── spec.md                    # Detailed specification
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── generation/           # Per-method generation configs
│   ├── training/             # YOLO training configs
│   └── evaluation/           # Metrics configs
├── src/
│   ├── data/                 # Dataset classes, loaders
│   ├── generation/           # Generator implementations
│   │   ├── base.py
│   │   ├── copy_paste.py
│   │   ├── inpainting.py
│   │   ├── controlnet.py
│   │   ├── textual_inversion.py
│   │   └── lora.py
│   ├── training/             # YOLO training wrappers
│   ├── evaluation/           # Metrics, visualization
│   └── utils/                # Common utilities
├── scripts/
│   ├── generate.py           # Batch generation script
│   ├── train_yolo.py         # Training runner
│   ├── evaluate.py           # Evaluation runner
│   └── compare.py            # Generate comparison report
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_generation_demo.ipynb
│   └── 03_results_analysis.ipynb
├── experiments/              # Experiment outputs
│   ├── baseline/
│   ├── inpainting/
│   ├── controlnet/
│   └── ...
└── data/                     # Symlink or copy of dataset
    ├── raw/
    ├── processed/
    └── synthetic/
```

---

---

# Detailed Specifications

## 1. Data Pipeline Specification

### 1.1 Input Data Format

**Directory Structure**:
```
data/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   │   ├── img_001.txt      # YOLO format annotations
│   │   ├── img_002.txt
│   │   └── ...
│   ├── val/
│   └── test/
└── classes.txt               # Class names, one per line
```

**YOLO Annotation Format** (`labels/train/img_001.txt`):
```
<class_id> <x_center> <y_center> <width> <height>
0 0.453 0.621 0.102 0.089
2 0.712 0.334 0.056 0.045
```
- All values normalized to [0, 1]
- One line per bounding box

**classes.txt Example**:
```
scratch
crack
dent
stain
discoloration
missing_component
deformation
contamination
...
```

### 1.2 Data Split Strategy

| Split | Purpose | Composition |
|-------|---------|-------------|
| Train | Model training | 70% of real data + synthetic |
| Val | Hyperparameter tuning | 15% of real data only |
| Test | Final evaluation | 15% of real data only |

**Critical**: Validation and test sets contain ONLY real data to ensure fair comparison.

### 1.3 Dataset Class Interface

```python
class DefectDataset:
    """Base dataset class for defect detection."""

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        classes_file: Path,
        transform: Optional[Callable] = None
    ):
        ...

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[BBox]]:
        """Return image and list of bounding boxes."""
        ...

    def get_class_distribution(self) -> Dict[str, int]:
        """Return count of instances per class."""
        ...


class SyntheticDefectDataset(DefectDataset):
    """Dataset that combines real and synthetic data."""

    def __init__(
        self,
        real_dataset: DefectDataset,
        synthetic_dir: Path,
        synthetic_ratio: float = 1.0,  # 1.0 = equal real:synthetic
        quality_filter: Optional[Callable] = None
    ):
        ...
```

---

## 2. Generation Pipeline Specification

### 2.1 Base Generator Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class GeneratedSample:
    """Output from a generator."""
    image: np.ndarray                    # RGB image
    bboxes: List[Tuple[int, float, float, float, float]]  # (class_id, x, y, w, h)
    metadata: dict                       # Generation parameters, seed, etc.

class BaseGenerator(ABC):
    """Abstract base class for all generators."""

    def __init__(self, config: dict, device: str = "mps"):
        self.config = config
        self.device = device

    @abstractmethod
    def setup(self) -> None:
        """Load models, initialize pipelines."""
        pass

    @abstractmethod
    def generate(
        self,
        base_image: Optional[np.ndarray] = None,
        defect_class: Optional[str] = None,
        num_samples: int = 1,
        seed: Optional[int] = None
    ) -> List[GeneratedSample]:
        """Generate synthetic defect images."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources."""
        pass
```

### 2.2 Generator Implementations

#### 2.2.1 Copy-Paste Generator

```python
class CopyPasteGenerator(BaseGenerator):
    """
    Extracts defect regions and pastes onto good samples.

    Config:
        defect_images_dir: Path to images with defects
        good_images_dir: Path to defect-free images
        blend_mode: "poisson" | "alpha" | "direct"
        scale_range: (0.5, 2.0)  # Random scaling
        rotation_range: (-45, 45)  # Random rotation degrees
    """
```

**Annotation Strategy**: Track paste location → automatic bbox

#### 2.2.2 SD Inpainting Generator

```python
class InpaintingGenerator(BaseGenerator):
    """
    Uses Stable Diffusion inpainting to add defects.

    Config:
        model_id: "stabilityai/stable-diffusion-2-inpainting"
        prompts: Dict[str, str]  # class_name -> prompt template
        mask_strategy: "random" | "grid" | "edge_aware"
        mask_size_range: (0.05, 0.2)  # Fraction of image
        guidance_scale: 7.5
        num_inference_steps: 30
    """
```

**Prompt Templates Example**:
```yaml
prompts:
  scratch: "industrial surface with visible scratch mark, defect, damage"
  crack: "metal surface with crack, fracture, structural damage"
  stain: "surface contamination, stain, discoloration, dirty spot"
```

**Annotation Strategy**: Convert inpaint mask → bbox

#### 2.2.3 ControlNet Generator

```python
class ControlNetGenerator(BaseGenerator):
    """
    Uses ControlNet for structure-preserving generation.

    Config:
        model_id: "runwayml/stable-diffusion-v1-5"
        controlnet_id: "lllyasviel/sd-controlnet-canny"
        conditioning: "canny" | "depth" | "seg"
        prompts: Dict[str, str]
        guidance_scale: 7.5
        controlnet_conditioning_scale: 1.0
    """
```

**Annotation Strategy**: Use conditioning region → infer bbox

#### 2.2.4 Textual Inversion Generator

```python
class TextualInversionGenerator(BaseGenerator):
    """
    Uses learned concept tokens for defect generation.

    Config:
        model_id: "runwayml/stable-diffusion-v1-5"
        embeddings_dir: Path to trained embeddings
        token_map: Dict[str, str]  # class_name -> token
    """
```

**Training Config**:
```yaml
textual_inversion:
  learning_rate: 5e-4
  max_train_steps: 3000
  save_steps: 500
  gradient_accumulation_steps: 4
  mixed_precision: "no"  # MPS doesn't support fp16 well
```

#### 2.2.5 LoRA Generator

```python
class LoRAGenerator(BaseGenerator):
    """
    Uses LoRA fine-tuned models for generation.

    Config:
        model_id: "runwayml/stable-diffusion-v1-5"
        lora_weights_dir: Path to trained LoRA weights
        lora_scale: 0.8
    """
```

**Training Config**:
```yaml
lora:
  rank: 4  # Low rank for MPS, increase for cloud
  alpha: 4
  learning_rate: 1e-4
  max_train_steps: 1000
  gradient_accumulation_steps: 4
  use_8bit_adam: false  # Not supported on MPS
```

### 2.3 Batch Generation Script Interface

```bash
# Generate synthetic data using a specific method
python scripts/generate.py \
    --method inpainting \
    --config configs/generation/inpainting.yaml \
    --good-images data/good_samples/ \
    --output-dir data/synthetic/inpainting/ \
    --num-per-class 500 \
    --seed 42

# Output structure:
# data/synthetic/inpainting/
# ├── images/
# │   ├── synth_0001.jpg
# │   └── ...
# ├── labels/
# │   ├── synth_0001.txt
# │   └── ...
# └── generation_log.json  # Metadata for reproducibility
```

---

## 3. Training Pipeline Specification

### 3.1 YOLO Training Configuration

```yaml
# configs/training/yolo_base.yaml
model: yolov8m.pt  # Medium model, balance speed/accuracy
data: data.yaml
epochs: 100
imgsz: 640
batch: 16  # Adjust for MPS memory
device: mps
patience: 20
save_period: 10

# Augmentation (traditional, applied during training)
augment: true
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 10
translate: 0.1
scale: 0.5
fliplr: 0.5
mosaic: 1.0
```

### 3.2 Experiment Naming Convention

```
experiments/
├── exp_001_baseline_real_only/
├── exp_002_traditional_aug/
├── exp_003_copypaste/
├── exp_004_inpainting_ratio_0.5/
├── exp_005_inpainting_ratio_1.0/
├── exp_006_inpainting_ratio_2.0/
├── exp_007_controlnet/
├── exp_008_textual_inversion/
├── exp_009_lora_rank4/
└── exp_010_best_ensemble/
```

**Experiment Metadata** (`experiment.json`):
```json
{
  "name": "exp_004_inpainting_ratio_0.5",
  "method": "inpainting",
  "synthetic_ratio": 0.5,
  "real_samples": 1000,
  "synthetic_samples": 500,
  "generation_config": "configs/generation/inpainting.yaml",
  "training_config": "configs/training/yolo_base.yaml",
  "timestamp": "2026-01-06T10:30:00Z",
  "git_hash": "abc123"
}
```

### 3.3 Training Script Interface

```bash
python scripts/train_yolo.py \
    --experiment-name exp_004_inpainting_ratio_0.5 \
    --real-data data/processed/ \
    --synthetic-data data/synthetic/inpainting/ \
    --synthetic-ratio 0.5 \
    --config configs/training/yolo_base.yaml \
    --output-dir experiments/
```

---

## 4. Evaluation Pipeline Specification

### 4.1 Metrics

#### Generation Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| FID | Fréchet Inception Distance to real defects | < 50 |
| LPIPS | Perceptual similarity | < 0.3 |
| Visual Score | Expert review (1-5 scale) | > 3.5 |

#### Detection Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| mAP@0.5 | Mean AP at IoU 0.5 | Baseline + 5% |
| mAP@0.5:0.95 | Mean AP across IoU thresholds | Baseline + 3% |
| Per-class AP | AP for each defect class | All > 0.5 |
| Precision | True positives / predictions | > 0.8 |
| Recall | True positives / ground truth | > 0.8 |

### 4.2 Evaluation Script Interface

```bash
python scripts/evaluate.py \
    --experiment-dir experiments/exp_004_inpainting_ratio_0.5/ \
    --test-data data/processed/test/ \
    --output-file experiments/exp_004_inpainting_ratio_0.5/results.json

# Output (results.json):
{
  "detection": {
    "mAP_50": 0.856,
    "mAP_50_95": 0.723,
    "per_class_ap": {
      "scratch": 0.891,
      "crack": 0.812,
      ...
    },
    "precision": 0.847,
    "recall": 0.823
  },
  "generation": {
    "fid": 42.3,
    "lpips": 0.24
  }
}
```

### 4.3 Comparison Report

```bash
python scripts/compare.py \
    --experiments-dir experiments/ \
    --output report.md

# Generates comparison table and charts
```

**Sample Output Table**:
```
| Experiment | Method | Synth Ratio | mAP@0.5 | Δ vs Baseline |
|------------|--------|-------------|---------|---------------|
| exp_001 | baseline | 0.0 | 0.756 | - |
| exp_002 | trad_aug | 0.0 | 0.782 | +3.4% |
| exp_003 | copypaste | 1.0 | 0.801 | +6.0% |
| exp_004 | inpainting | 0.5 | 0.823 | +8.9% |
| exp_005 | inpainting | 1.0 | 0.856 | +13.2% |
| exp_006 | inpainting | 2.0 | 0.841 | +11.2% |
| exp_007 | controlnet | 1.0 | 0.867 | +14.7% |
```

---

## 5. Success Criteria

### Primary Goals
- [ ] Synthetic data improves mAP@0.5 by at least **+5%** over baseline
- [ ] Per-class AP improves for at least **80%** of defect classes
- [ ] Generated samples pass visual quality review (score > 3.5/5)

### Secondary Goals
- [ ] Identify best generation method for this dataset
- [ ] Determine optimal synthetic-to-real ratio
- [ ] Document failure cases and limitations

### Deliverables
1. **Generation Models**: Trained generators (if applicable) for each method
2. **Synthetic Datasets**: Generated images + annotations for best methods
3. **Trained Detectors**: YOLO checkpoints for each experiment
4. **Comparison Report**: Comprehensive metrics and visualizations
5. **Recommendations**: Best approach for production deployment

---

## 6. Dependencies

### requirements.txt
```
# Core
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
Pillow>=9.0.0
opencv-python>=4.7.0

# Diffusion
diffusers>=0.25.0
transformers>=4.35.0
accelerate>=0.25.0
peft>=0.7.0  # For LoRA

# Detection
ultralytics>=8.0.0

# Augmentation
albumentations>=1.3.0

# Metrics
torchmetrics>=1.0.0
clean-fid>=0.1.35
lpips>=0.1.4

# Experiment tracking
wandb>=0.16.0

# Utilities
pyyaml>=6.0
tqdm>=4.65.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 7. Execution Milestones

### Milestone 1: Setup & Baseline
**Objective**: Establish project foundation and baseline performance
1. Create project structure and install dependencies
2. Prepare data in required format (YOLO annotations)
3. Train baseline YOLO (no synthetic data)
4. Train with traditional augmentation
5. Record baseline metrics

**Exit Criteria**: Baseline mAP recorded, data pipeline working

### Milestone 2: Non-Training Generation Methods
**Objective**: Implement inference-only synthetic data generation
6. Implement and run Copy-Paste generation
7. Implement and run SD Inpainting generation
8. Implement and run ControlNet generation
9. Validate generated samples visually

**Exit Criteria**: 3 synthetic datasets generated with annotations

### Milestone 3: YOLO Training on Synthetic Data
**Objective**: Evaluate impact of each generation method
10. Train YOLO on Copy-Paste augmented data
11. Train YOLO on Inpainting augmented data
12. Train YOLO on ControlNet augmented data
13. Test different synthetic-to-real ratios (0.5, 1.0, 2.0)

**Exit Criteria**: Detection models trained for each method/ratio combination

### Milestone 4: Training-Based Methods (Optional)
**Objective**: Explore fine-tuned generation methods
14. Train Textual Inversion (per defect class)
15. Train LoRA (per defect class)
16. Generate samples with trained models
17. Train YOLO on these datasets

**Exit Criteria**: Fine-tuned generators and corresponding detection models
**Note**: Skip if Milestone 3 results are satisfactory

### Milestone 5: VLM Evaluation
**Objective**: Evaluate VLMs with real and synthetic data
18. Evaluate Gemini API in zero-shot mode on test set
19. Evaluate Gemini with few-shot prompting (real examples)
20. Evaluate Gemini with few-shot prompting (synthetic examples)
21. Fine-tune Qwen-VL with LoRA on real data
22. Fine-tune Qwen-VL with LoRA on real + synthetic data
23. Parse VLM outputs (`<think>`, `<final>` tags) to bboxes

**Exit Criteria**: VLM detection metrics comparable to YOLO

### Milestone 6: Analysis & Reporting
**Objective**: Compare all methods and document findings
24. Compute all metrics (FID, mAP, per-class AP) for all models
25. Generate comparison report: YOLO vs VLMs, Real vs Synthetic
26. Identify best generation method and optimal ratio
27. Document recommendations, failure cases, and limitations

**Exit Criteria**: Final report with actionable recommendations

---

## VLM Evaluation Specification

### Gemini API Evaluation

**Zero-Shot Prompt**:
```
Analyze this industrial inspection image for defects.
For each defect found, provide:
1. Defect type (from: scratch, crack, dent, stain, ...)
2. Bounding box coordinates [x_min, y_min, x_max, y_max]

Format your response as:
<think>
[Your reasoning about what you see]
</think>
<final>
[{"type": "defect_class", "bbox": [x1, y1, x2, y2]}, ...]
</final>
```

**Few-Shot Prompt**:
```
Here are examples of industrial defects:
[Example 1: image + annotation]
[Example 2: image + annotation]
...

Now analyze this new image:
[Test image]

<think>...</think>
<final>...</final>
```

### Qwen-VL LoRA Fine-tuning

**Training Configuration**:
```yaml
model: Qwen/Qwen-VL-Chat
lora:
  r: 8
  alpha: 16
  target_modules: ["c_attn", "attn.c_proj", "w1", "w2"]
  dropout: 0.05
training:
  batch_size: 2  # Limited by 16GB memory
  gradient_accumulation: 8
  learning_rate: 1e-4
  epochs: 3
  fp16: false  # MPS compatibility
```

### VLM Output Parsing

```python
import re
import json

def parse_vlm_output(response: str) -> List[Dict]:
    """Parse VLM structured output to bounding boxes."""
    # Extract <final> content
    final_match = re.search(r'<final>(.*?)</final>', response, re.DOTALL)
    if not final_match:
        return []

    try:
        detections = json.loads(final_match.group(1))
        return [
            {
                "class": d["type"],
                "bbox": d["bbox"],  # [x1, y1, x2, y2]
                "confidence": d.get("confidence", 1.0)
            }
            for d in detections
        ]
    except json.JSONDecodeError:
        return []
```

### VLM vs YOLO Comparison Matrix

| Aspect | YOLO v11 | Qwen-VL | Gemini |
|--------|----------|---------|--------|
| Training | Full fine-tune | LoRA | None (API) |
| Inference | Local (fast) | Local (slow) | API (rate-limited) |
| Few-shot | N/A | Supported | Supported |
| Output | Native bbox | Structured text | Structured text |
| Cost | Compute only | Compute only | API credits |
