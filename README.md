# Assessing Pure Temporal Perception in Large Vision-Language Models

> **Investigating the capabilities of Video LLMs to perceive motion-defined objects absent of spatial cues.**

![Period](https://img.shields.io/badge/Period-2025.11_--_2026.02-blue) ![Model](https://img.shields.io/badge/Model-Qwen_VL-violet) ![Status](https://img.shields.io/badge/Status-Research-green)

## Abstract

Current state-of-the-art Video Large Language Models (Video LLMs) typically process video inputs as a sequence of static images, relying heavily on spatial feature extraction (e.g., ViT, CLIP encoders) for each frame. This architecture raises a fundamental question regarding the true "temporal understanding" of these models:

> **"Can a Video LLM perceive information encoded solely in the temporal domain, without any spatial features available in individual frames?"**

To address this, this research introduces an experimental benchmark utilizing **"Motion-Defined Signal in Temporal Noise"** (often referred to as the "Unscreenshottable" phenomenon). In these synthesized videos, semantic information (text) is mathematically invisible in any single static frame and becomes perceivable only through the temporal integration of motion dynamics. This study aims to evaluate whether Qwen-VL and other VLLMs can perform Optical Character Recognition (OCR) by effectively integrating temporal information, similar to the human visual system's motion processing.

## Research Goals & Objectives

The primary goal is to evaluate and potentially expand the **temporal reasoning capabilities** of Video LLMs, with a specific focus on **Qwen-VL**, in scenarios where static visual cues are completely absent.

1. **Vision Research:** Analyze whether current VLLMs can perceive "Motion-Defined Objects" (structure-from-motion) without reliance on spatial edge detection.
2. **Benchmark Construction:** Develop a synthesized dataset of "noise-flow" videos containing text that is imperceptible via static analysis but visible via temporal correlation.
3. **Academic Contribution:** rigorous analysis of the findings for submission to computer vision conferences.

## Methodology: Data Synthesis Pipeline

To evaluate the model's temporal perception, we construct a synthetic dataset of **(Video, Text)** pairs. The core principle is to decouple spatial information from semantic content; the text is defined solely by the relative motion of noise patterns, not by pixel intensity or color contrast.

### 1. Generation Logic

The data generation process follows a frame-by-frame noise regeneration and transformation algorithm:

1. **Initialization ($t=0$):**
    A full-resolution noise frame $N_0$ is generated using a specified noise distribution (e.g., Gaussian, Uniform).
2. **Region Definition (Masking):**
    A binary mask $M$ is defined for the target text. The frame is divided into two independent regions:
    - **Signal Region ($R_{text}$):** Pixels corresponding to the text.
    - **Background Region ($R_{bg}$):** Pixels outside the text.
3. **Independent Transformation ($t \to t+1$):**
    Distinct motion functions are applied to each region:
    $$N_{t+1}[p] = \begin{cases} T_{text}(N_t[p]) & \text{if } p \in R_{text} \\ T_{bg}(N_t[p]) & \text{if } p \in R_{bg} \end{cases}$$
4. **Stochastic In-filling:**
    When pixels shift due to transformation $T$, the newly exposed "void" areas are filled with **freshly generated noise**. This ensures that every individual frame remains statistically indistinguishable from random noise, preserving the "unscreenshottable" property.

### 2. Configuration Space (Metadata)

The dataset is controlled by the following hyperparameters:

- **Noise Properties:**
  - *Generation Method:* (TBC; e.g. Gaussian / Uniform / Salt-and-Pepper)
  - *Depth (Resolution):* $H \times W$ (e.g., 224x224, 512x512)
- **Temporal Properties:**
  - *Video Duration:* Total time ($s$)
  - *Frame Count:* Total frames ($F$)
  - *Human Captuability:* True/False

### 3. Motion Primitives ($T$)

Both the **Signal Region ($R_{text}$)** and **Background Region ($R_{bg}$)** can be assigned one of the following motion dynamics independently:

| Motion Type | Description |
| :--- | :--- |
| **Static** | The noise pattern remains fixed (frozen). |
| **Linear** | Translation along cardinal axes: **Up / Down / Left / Right**. |
| **Rotational** | Angular shift around the center: **Clockwise / Counter-Clockwise**. |
| **Radial** | Spatial scaling towards a central point: **Convergence (Pinch-in)**. |

### 4. Experimental Scenarios

By combining the motion primitives of the text and background, we simulate various difficulty levels for temporal object detection:

- *Ex:* Text (Linear Right) vs. Background (Static)
- *Ex:* Text (Static) vs. Background (Linear Left)
- *Ex:* Text (Clockwise) vs. Background (Counter-Clockwise)

## Project Information

- **Project Duration:** 2025.11.18 – 2026.02
- **Affiliation:** Department of Artificial Intelligence, Undergraduate Research

### Research Team

| Role | Name |
| :--- | :--- |
| **Project Leader** | Joo-heon Kang, Hong-seok Oh |
| **Researcher** | **Jae-hyun Ahn** |

## Repository Structure

- `code/`: TBC
- `data/`: TBC
- `scripts/`: TBC

## Getting Started

> *(Instructions for environment setup, dependency installation, and data generation will be updated.)*

---
*This project is conducted as an undergraduate research initiative aimed at exploring the frontiers of Temporal Video Understanding.*

*This Document have written with Gemini 2.5 Think.*
