# Generative-AI-for-New-Diseases-Imaging-Biomarker-Discovery
# Conditional Diffusion for Lung CT Synthesis and Biomarker Exploration

This repository contains the code developed for an academic research project on
synthetic lung CT image generation using diffusion models with anatomical conditioning.

The project focuses on:
- Controlled generation of 2D lung CT slices
- Conditioning on slice depth (z-position)
- Evaluation of synthetic image quality
- Exploratory biomarker analysis on real and synthetic data

This work is intended as a **research baseline**, not a clinical system.

---

## Project Overview

Diffusion models have recently shown strong performance in image generation.
In medical imaging, they offer the potential to generate anatomically consistent
synthetic data for research and augmentation purposes.

In this project, diffusion models are trained to generate lung CT slices under
different conditioning strategies:
- Unconditional diffusion
- Continuous conditioning on normalized slice depth
- Discrete z-bin conditioning
- Classifier-Free Guidance (CFG)

The generated images are analyzed both visually and through exploratory
explainability techniques to study whether meaningful anatomical patterns are preserved.

---

## Dataset

- Input data consists of 2D lung CT slices stored as `.npy` files
- Images are normalized to the range `[0, 1]` and resized to `128×128` or `256×256`
- Slice depth information is extracted from file names and used for conditioning
- Discrete z-bins are defined to group slices by anatomical depth

⚠️ The dataset is not included in this repository.

---

## Model Architecture

- U-Net–based diffusion models
- Implementations using:
  - Custom DDPM code
  - HuggingFace `diffusers` library
- Noise schedules:
  - Linear and cosine (squared cosine)
- Sampling:
  - DDPM
  - DDIM (deterministic and stochastic)

Conditioning mechanisms:
- Continuous scalar conditioning (z ∈ [0, 1])
- Discrete class conditioning (z-bins)
- Classifier-Free Guidance (CFG)

---

## Training Strategy

- Mean Squared Error loss on noise prediction
- Exponential Moving Average (EMA) of model weights
- Fixed-noise DDIM sampling for stable visual evaluation
- Multiple training regimes to study stability and conditioning effects

Training scripts are located in the `scripts/` directory.

---

## Evaluation

### Visual Evaluation
- Progressive improvement over training steps
- Best results obtained for mid and upper lung regions
- Failure cases include:
  - Anatomical distortions
  - Residual noise
  - Lower quality in lower lung slices

### Quantitative / Proxy Evaluation
- Fixed-noise comparison across checkpoints
- Internal consistency metrics
- Metrics are used as **supportive signals**, not clinical validation

---

## Biomarker Exploration

An exploratory analysis was conducted to study whether attribution methods
highlight meaningful regions in real and synthetic CT images.

Findings:
- Attention often focuses on high-contrast structures (edges, bones, boundaries)
- Limited localization inside lung parenchyma
- Highlights the need for anatomical constraints and segmentation

This analysis is **exploratory** and not intended for diagnosis.

---

## Repository Structure

```text
.
├── src/                # Core diffusion models and utilities
├── scripts/            # Training and generation scripts
├── data/               # Dataset (not included)
├── outputs/            # Generated samples and checkpoints
├── requirements.txt
└── README.md
