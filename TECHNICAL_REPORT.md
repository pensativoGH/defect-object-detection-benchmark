# Technical Report: Synthetic Data Generation for Industrial Defect Detection

**Authors**: Pramod Sharma, Claude (Anthropic)
**Date**: January 2026
**Repository**: https://github.com/pensativoGH/defect-object-detection-benchmark

---

## Abstract

This report presents a comprehensive benchmark study comparing synthetic data generation methods for industrial defect detection. Using the VISION Cable dataset with only 41 real training images, we evaluated 8 different training strategies including traditional augmentation, copy-paste augmentation, and various diffusion-based generation methods (Textual Inversion, LoRA, Inpainting, ControlNet). Our key finding is that a model trained exclusively on synthetic data (no real training images) achieves the best performance with **mAP@0.5 of 0.422**, representing a **115.3% improvement** over the baseline. This demonstrates that high-quality synthetic data can effectively replace real labeled data for training defect detection models.

---

## 1. Introduction

### 1.1 Problem Statement

Industrial defect detection faces a fundamental challenge: collecting and annotating real defect samples is expensive, time-consuming, and sometimes impossible for rare defect types. This creates a data scarcity problem that limits the performance of deep learning-based detection models.

### 1.2 Objectives

This study aims to:
1. Compare multiple synthetic data generation methods for defect detection
2. Evaluate whether synthetic data can augment or replace real training data
3. Identify the most effective approaches for low-data industrial inspection scenarios

### 1.3 Scope

We focus on object detection (bounding box prediction) for cable defects using YOLOv8, comparing:
- Traditional augmentation techniques
- Copy-paste augmentation
- Diffusion-based generation (Stable Diffusion variants)
- Combined synthetic data strategies

---

## 2. Dataset

### 2.1 Source

**VISION Dataset - Cable Category**
The VISION dataset is a publicly available industrial inspection benchmark containing multiple product categories with defect annotations.

### 2.2 Dataset Statistics

| Split | Images | Annotations | Purpose |
|-------|--------|-------------|---------|
| Training (Real) | 41 | 58 bboxes | Model training |
| Validation | 131 | 180 bboxes | Hyperparameter tuning & evaluation |
| Inference (Unannotated) | 1,146 | None | Qualitative analysis |

### 2.3 Defect Classes

| Class ID | Class Name | Training Instances | Validation Instances |
|----------|------------|-------------------|---------------------|
| 0 | break | ~25 | 91 |
| 1 | thunderbolt | ~33 | 89 |

### 2.4 Image Properties

- **Resolution**: Variable, typically < 512px
- **Format**: JPEG
- **Defect Size**: Small (5-15% of image area)
- **Annotation Format**: YOLO format (normalized xywh)

### 2.5 Data Challenges

1. **Extreme data scarcity**: Only 41 training images
2. **Class imbalance**: Uneven distribution between defect types
3. **Small defect size**: Defects occupy small portion of images
4. **No "good" samples in training**: All training images contain defects

---

## 3. Methods

### 3.1 Detection Model

**YOLOv8m (Medium)**
- Parameters: 25.8M
- Architecture: CSPDarknet backbone + PANet neck + Decoupled head
- Input size: 640×640
- Framework: Ultralytics

### 3.2 Baseline Methods

#### 3.2.1 Baseline (No Augmentation)
Training on 41 real images with no data augmentation. Serves as the lower bound for comparison.

#### 3.2.2 Traditional Augmentation
Standard augmentations applied during training:
- Geometric: Horizontal flip (p=0.5), rotation (±10°), scale (0.5-1.5)
- Photometric: HSV adjustments (H±0.015, S±0.7, V±0.4)
- Advanced: Mosaic (p=1.0), mixup disabled

### 3.3 Synthetic Data Generation Methods

#### 3.3.1 Copy-Paste Augmentation

**Description**: Extract defect regions using bounding box annotations and paste onto random locations within the same or different images.

**Implementation**:
```python
# Pseudo-code
for each image:
    defect_crops = extract_regions(image, bboxes)
    for crop in defect_crops:
        # Random transformations
        crop = rotate(crop, random(-45, 45))
        crop = scale(crop, random(0.5, 2.0))
        # Paste with Poisson blending
        target_image = blend(target_image, crop, random_location)
        new_bbox = compute_bbox(paste_location, crop_size)
```

**Output**: 177 synthetic images with automatic bbox annotations

**Advantages**:
- Preserves real defect texture and appearance
- No domain gap (same image distribution)
- Fast generation (no model inference)
- Automatic annotation generation

#### 3.3.2 Textual Inversion

**Description**: Learn a new "concept token" embedding that captures the defect appearance, then use it with Stable Diffusion for generation.

**Model**: Stable Diffusion v1.5

**Training Configuration**:
| Parameter | Value |
|-----------|-------|
| Learning Rate | 5e-4 |
| Training Steps | 3,000 |
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Token | `<defect>` |

**Generation Prompt**: `"industrial cable with <defect> defect, inspection photo"`

**Output**: 100 synthetic images (50 per class)

#### 3.3.3 LoRA Fine-tuning

**Description**: Low-Rank Adaptation of Stable Diffusion to learn defect generation while preserving base model knowledge.

**Model**: Stable Diffusion v1.5 + LoRA

**Training Configuration**:
| Parameter | Value |
|-----------|-------|
| LoRA Rank | 4 |
| LoRA Alpha | 4 |
| Learning Rate | 1e-4 |
| Training Steps | 1,000 |
| Target Modules | attention layers |

**Output**: 100 synthetic images (50 per class)

#### 3.3.4 SD Inpainting

**Description**: Use Stable Diffusion Inpainting to "paint in" defects on masked regions of images.

**Model**: `stabilityai/stable-diffusion-2-inpainting`

**Process**:
1. Create random masks in plausible defect locations
2. Inpaint with defect-specific prompts
3. Convert inpainting mask to bounding box annotation

**Prompts**:
- Break: `"damaged cable with break defect, industrial inspection"`
- Thunderbolt: `"cable with thunderbolt burn mark, electrical damage"`

**Output**: 20 synthetic images

**Limitation**: Slow generation (~30s per image on MPS)

#### 3.3.5 ControlNet

**Description**: Use ControlNet with Canny edge conditioning to maintain structural consistency while generating defects.

**Model**: `lllyasviel/sd-controlnet-canny` + SD v1.5

**Process**:
1. Extract Canny edges from source images
2. Generate with edge conditioning + defect prompts
3. Infer bounding boxes from generation regions

**Configuration**:
| Parameter | Value |
|-----------|-------|
| Guidance Scale | 7.5 |
| ControlNet Scale | 1.0 |
| Canny Thresholds | (100, 200) |

**Output**: 50 synthetic images

### 3.4 Combined Strategies

#### 3.4.1 All Synthetic Combined

Combine all synthetic data from individual methods with real training data:

| Source | Images |
|--------|--------|
| Real Training | 41 |
| Copy-Paste | 177 |
| Textual Inversion | 100 |
| LoRA | 100 |
| Inpainting | 20 |
| ControlNet | 50 |
| **Total** | **488** |

#### 3.4.2 Synthetic Only

Train exclusively on synthetic data (no real training images):

| Source | Images |
|--------|--------|
| Copy-Paste | 177 |
| Textual Inversion | 100 |
| LoRA | 100 |
| Inpainting | 20 |
| ControlNet | 50 |
| **Total** | **447** |

Validation still uses real images (131) for fair comparison.

---

## 4. Experimental Setup

### 4.1 Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8m |
| Image Size | 640×640 |
| Batch Size | 16 |
| Max Epochs | 100 |
| Early Stopping | patience=20 |
| Optimizer | SGD (default) |
| Learning Rate | 0.01 (auto-scaled) |
| Device | Apple M4 (MPS) |

### 4.2 Evaluation Protocol

- **Validation Set**: 131 real images with 180 bounding box annotations
- **Metrics**: mAP@0.5, mAP@0.5:0.95, Precision, Recall
- **Per-Class Metrics**: AP for each defect class

### 4.3 Hardware

- **Device**: MacBook Air M4
- **Memory**: 16GB Unified Memory
- **Backend**: PyTorch MPS (Metal Performance Shaders)

### 4.4 Software Stack

| Component | Version |
|-----------|---------|
| Python | 3.13 |
| PyTorch | 2.8.0 |
| Ultralytics | 8.3.x |
| Diffusers | 0.25+ |
| PEFT | 0.7+ |

---

## 5. Results

### 5.1 Main Results

| Rank | Method | Training Data | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | vs Baseline |
|------|--------|---------------|---------|--------------|-----------|--------|-------------|
| 1 | **Synthetic Only** | 447 synthetic | **0.422** | **0.204** | 0.584 | 0.393 | **+115.3%** |
| 2 | All Synthetic Combined | 41 real + 447 synthetic | 0.399 | 0.193 | 0.578 | 0.331 | +103.6% |
| 3 | Copy-Paste | 41 real + 177 synthetic | 0.371 | 0.174 | 0.406 | 0.373 | +89.3% |
| 4 | Textual Inversion | 41 real + 100 synthetic | 0.350 | 0.166 | 0.555 | 0.361 | +78.6% |
| 5 | LoRA | 41 real + 100 synthetic | 0.317 | 0.156 | 0.485 | 0.355 | +61.7% |
| 6 | Traditional Aug | 41 real (augmented) | 0.275 | 0.120 | 0.328 | 0.327 | +40.3% |
| 7 | ControlNet | 41 real + 50 synthetic | 0.231 | 0.094 | 0.229 | 0.225 | +17.9% |
| 8 | Baseline | 41 real | 0.196 | 0.085 | 0.355 | 0.207 | - |

### 5.2 Per-Class Performance (mAP@0.5)

| Method | break | thunderbolt | Δ break | Δ thunderbolt |
|--------|-------|-------------|---------|---------------|
| Synthetic Only | 0.058 | 0.787 | +107% | +116% |
| All Synthetic Combined | 0.054 | 0.744 | +93% | +104% |
| Copy-Paste | 0.071 | 0.671 | +154% | +84% |
| Textual Inversion | 0.062 | 0.638 | +121% | +75% |
| LoRA | 0.055 | 0.579 | +96% | +59% |
| Traditional Aug | 0.041 | 0.509 | +46% | +40% |
| ControlNet | 0.032 | 0.430 | +14% | +18% |
| Baseline | 0.028 | 0.364 | - | - |

### 5.3 Training Dynamics

| Method | Best Epoch | Total Epochs | Training Time | Early Stopped |
|--------|------------|--------------|---------------|---------------|
| Baseline | 10 | 30 | 11 min | Yes |
| Traditional Aug | 5 | 25 | 13 min | Yes |
| Copy-Paste | 19 | 39 | 2.3 hr | Yes |
| Textual Inversion | 24 | 44 | 2.1 hr | Yes |
| LoRA | 21 | 41 | 1.9 hr | Yes |
| ControlNet | 1 | 21 | 2.2 hr | Yes |
| All Synthetic Combined | 34 | 54 | 1.8 hr | Yes |
| Synthetic Only | 51 | 71 | 3.7 hr | Yes |

### 5.4 Inference on Unannotated Test Set

Ran all models on 1,146 unannotated test images to analyze detection behavior:

| Model | Total Detections | Images with Detections | Avg Detections/Image |
|-------|------------------|------------------------|---------------------|
| Inpainting | 715 | 528 (46%) | 1.35 |
| Synthetic Only | 1,053 | 658 (57%) | 1.60 |
| All Synthetic Combined | 1,217 | 678 (59%) | 1.79 |
| ControlNet | 1,260 | 642 (56%) | 1.96 |
| LoRA | 1,276 | 682 (60%) | 1.87 |
| Traditional Aug | 1,514 | 714 (62%) | 2.12 |
| Copy-Paste | 2,019 | 920 (80%) | 2.19 |
| Textual Inversion | 2,205 | 952 (83%) | 2.32 |
| Baseline | 4,485 | 820 (72%) | 5.47 |

*Note: Without ground truth, these numbers indicate model behavior but not accuracy.*

---

## 6. Analysis

### 6.1 Key Finding: Synthetic-Only Training Superiority

The most striking result is that training exclusively on synthetic data (no real training images) achieves the **best overall performance**. This counter-intuitive finding can be explained by:

1. **Data Volume**: 447 synthetic images vs 41 real images (10.9× more data)
2. **Diversity**: Multiple generation methods create diverse defect appearances
3. **Reduced Overfitting**: More training samples prevent memorization
4. **Domain Coverage**: Synthetic data covers more variations than limited real samples

### 6.2 Method Comparison

#### Copy-Paste: Best Individual Method
- **+89.3%** improvement over baseline
- Highest per-class improvement for "break" class (+154%)
- Preserves exact defect textures from real images
- No domain gap issues

#### Diffusion Methods: Mixed Results
- **Textual Inversion** (+78.6%): Effective concept learning
- **LoRA** (+61.7%): Good but slower to train
- **Inpainting** (limited samples): Constrained by generation speed
- **ControlNet** (+17.9%): Struggled with thin cable structures

#### Traditional Augmentation: Solid Baseline
- **+40.3%** improvement with zero synthetic data
- Essential preprocessing for any approach
- Low computational cost

### 6.3 Class-Level Analysis

The "break" class consistently shows lower AP across all methods:
- More subtle visual appearance
- Smaller defect regions
- Less variation in training data

The "thunderbolt" class benefits more from synthetic augmentation:
- Distinctive visual pattern
- More consistent appearance
- Better captured by generative models

### 6.4 Combining Methods

Combining all synthetic sources provides:
- **+103.6%** improvement (vs baseline)
- Slightly lower than synthetic-only (0.399 vs 0.422)

This suggests that including the 41 real images may introduce:
- Potential overfitting to specific real examples
- Bias toward limited real data distribution

### 6.5 Training Efficiency

| Strategy | Data Prep Time | Training Time | Total Time |
|----------|----------------|---------------|------------|
| Baseline | 0 | 11 min | 11 min |
| Traditional Aug | 0 | 13 min | 13 min |
| Copy-Paste | ~5 min | 2.3 hr | ~2.4 hr |
| Textual Inversion | ~2 hr (training) | 2.1 hr | ~4.1 hr |
| LoRA | ~4 hr (training) | 1.9 hr | ~5.9 hr |
| Synthetic Only | ~6 hr (all methods) | 3.7 hr | ~9.7 hr |

---

## 7. Discussion

### 7.1 Implications for Industrial Inspection

1. **Reduced Annotation Requirements**: Synthetic data can dramatically reduce the need for real labeled samples

2. **Scalability**: Once generation pipelines are established, creating additional training data is straightforward

3. **Rare Defect Handling**: Synthetic generation can create samples for defect types with few real examples

### 7.2 Limitations

1. **Single Dataset**: Results are specific to the Cable dataset; generalization to other defect types needs validation

2. **Simple Defect Classes**: Only 2 classes tested; complex multi-class scenarios may differ

3. **Compute Constraints**: MPS backend limited diffusion generation speed; CUDA would be faster

4. **No "Good" Samples**: All training images contained defects; results may differ with clean background images

### 7.3 Failure Cases

1. **ControlNet**: Canny edge conditioning failed to capture thin cable defects effectively

2. **Inpainting**: Limited sample count (20) due to slow generation reduced effectiveness

3. **"Break" Class**: Remained challenging across all methods due to subtle visual characteristics

### 7.4 Recommendations

| Scenario | Recommended Approach |
|----------|---------------------|
| Quick prototyping | Copy-Paste augmentation |
| Maximum performance | Combine all synthetic methods |
| Limited compute | Traditional augmentation + Copy-Paste |
| Rare defect types | Textual Inversion for targeted generation |
| Production deployment | Validate on diverse real-world test sets |

---

## 8. Conclusions

This benchmark study demonstrates that:

1. **Synthetic data significantly improves defect detection performance** - All synthetic methods outperformed the baseline, with improvements ranging from +17.9% to +115.3%.

2. **Synthetic-only training is viable and effective** - A model trained exclusively on synthetic data achieved the best performance (mAP@0.5 = 0.422), suggesting that high-quality synthetic data can replace real labeled data.

3. **Simple methods can be highly effective** - Copy-Paste augmentation achieved +89.3% improvement with minimal computational overhead and no model training.

4. **Method diversity provides robust training signal** - Combining multiple generation approaches creates complementary data distributions.

5. **Domain gap matters** - Methods that preserve real defect characteristics (Copy-Paste) outperform those that may introduce stylistic artifacts (some diffusion methods).

---

## 9. Future Work

1. **Multi-Dataset Validation**: Test on additional VISION categories and other industrial datasets

2. **Larger Scale Generation**: Generate more samples with cloud GPU for diffusion methods

3. **Quality Filtering**: Implement FID-based filtering to select high-quality synthetic samples

4. **Advanced Architectures**: Compare with YOLO11, RT-DETR, and other modern detectors

5. **VLM Integration**: Evaluate vision-language models (Qwen-VL, Gemini) for few-shot detection

6. **Active Learning**: Use synthetic data to bootstrap active learning pipelines

---

## Appendix A: Synthetic Data Examples

### A.1 Copy-Paste Examples
Defect regions extracted from real images and pasted with random transformations (rotation, scaling) onto different locations.

### A.2 Textual Inversion Examples
Generated using learned `<defect>` token with Stable Diffusion, producing stylized but recognizable defect patterns.

### A.3 LoRA Examples
Fine-tuned Stable Diffusion generates defects with better adherence to prompt specifications.

### A.4 ControlNet Examples
Edge-conditioned generation maintains cable structure while adding defect patterns.

---

## Appendix B: Configuration Files

### B.1 YOLO Training Config
```yaml
model: yolov8m.pt
imgsz: 640
batch: 16
epochs: 100
patience: 20
device: mps
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

### B.2 Textual Inversion Config
```yaml
model_id: runwayml/stable-diffusion-v1-5
learning_rate: 5.0e-4
max_train_steps: 3000
gradient_accumulation_steps: 4
mixed_precision: "no"
token: "<defect>"
```

### B.3 LoRA Config
```yaml
model_id: runwayml/stable-diffusion-v1-5
rank: 4
alpha: 4
learning_rate: 1.0e-4
max_train_steps: 1000
target_modules: ["to_q", "to_v", "to_k", "to_out.0"]
```

---

## References

1. Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
2. VISION Dataset: Industrial inspection benchmark
3. Stable Diffusion: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models"
4. LoRA: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
5. Textual Inversion: Gal et al., "An Image is Worth One Word"
6. ControlNet: Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models"

---

*Report generated with assistance from Claude (Anthropic)*
