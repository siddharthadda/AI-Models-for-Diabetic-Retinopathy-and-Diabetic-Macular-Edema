# AI Models for Diabetic Retinopathy & Diabetic Macular Edema

An advanced deep learning framework for the automated segmentation of retinal structures and classification of **Diabetic Retinopathy (DR)** and **Diabetic Macular Edema (DME)** severity stages from retinal fundus images.

This repository implements a multi-stage workflow:
1. **Segmentation Pipeline**: Employs state-of-the-art architectures (Attention U-Net, Residual Attention U-Net) to segment Blood Vessels (BV) and lesions (LS) from fundus images.
2. **Diagnostic Grading Pipeline**: Employs transfer learning (InceptionV3 and Attention-based networks) and ensemble stacking to grade the severity of DR and DME using the segmented structures as targeted inputs, significantly outperforming control models trained on raw images.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features & Architectures](#key-features--architectures)
- [Repository Structure](#repository-structure)
- [Use Cases](#use-cases)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Workflow Guide](#workflow-guide)
  - [1. Data Preprocessing & Segmentation](#1-data-preprocessing--segmentation)
  - [2. Diagnostic & Prognostic Grading](#2-diagnostic--prognostic-grading)
  - [3. Evaluation & Metrics Visualization](#3-evaluation--metrics-visualization)
- [Results & Performance](#results-performance)

---

## Project Overview

Diabetic Retinopathy (DR) and Diabetic Macular Edema (DME) are leading causes of vision loss worldwide. Manual diagnosis requires expert ophthalmologists to identify microaneurysms, hemorrhages, exudates, and abnormal blood vessel growth in retinal fundus photography.

This project addresses these challenges by developing a pipeline that isolates key diagnostic regions through deep segmentation networks before feeding them into grading classification networks:
- **Vessel-Based Diagnosis**: Segmenting blood vessels (BV) using Attention ResU-Net to diagnose vascular changes.
- **Lesion-Based Diagnosis**: Segmenting diagnostic lesions (LS) including microaneurysms, hemorrhages, and exudates.
- **Grading & Ensemble Stacking**: Utilizing segmented masks to train diagnostic models and applying ensemble techniques (Logistic Regression, Random Forests, SVMs) to consolidate predictions.

---

## Key Features & Architectures

### 1. Blood Vessel (RSEG) Segmentation
- **Architecture**: `Attention_ResUNet` and `residual_attentionunet` utilizing TensorFlow/Keras.
- **Loss Functions**: Customized Sørensen–Dice Coefficient Loss, Dice Coefficient optimization.
- **Image Preprocessing**: Integrated Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel in Lab color space to enhance blood vessel contrast.

### 2. Multi-class Lesion (MultiSeg) Segmentation
- **Architecture**: Attention U-Net and Bilateral U-Net (`BUNETFOCALNOWEIGHTS`) designed to identify multi-class lesions (microaneurysms, hemorrhages, and hard/soft exudates).
- **Loss Functions**: Focal Loss / Categorical Focal Dice Loss to handle extreme class imbalance between healthy tissue and tiny lesion clusters.

### 3. Classification & Ensemble Grading (DRGrading & DMEGrading)
- **Architecture**: InceptionV3 with attention gates (`inceptionattmodel`) mapping features to diagnostic levels.
- **Meta-Classification (Stacking)**: Blends multiple models using a stacked meta-learner (Logistic Regression/Random Forest/SVM) to output final diagnostic levels.

---

## Repository Structure

```
.
├── Demo/
│   └── Demo_image/
│       └── Tri.jpg                 # Demo target image for testing predictions
├── DRGrading/
│   ├── csv/                        # Training metrics/logs for DR diagnostic models
│   ├── saved_models/               # Saved model weights (e.g., BV quick_train fingerprint/saved_model.pb)
│   └── old_models/                 # Archive of legacy models (H5 format) and intermediate data CSVs
├── DMEGrading/
│   └── csv/                        # Metric sheets for DME Prognostic Model
├── Segmentation/
│   ├── RSEG/                       # Blood Vessel (BV) Segmentation code and models
│   │   ├── Unet_VesselSegmentation_preprocessing.ipynb
│   │   ├── Unet_VesselSegmentation_preprocessing-new.ipynb
│   │   └── AttResUNET.hdf5        # Trained weights for Attention ResU-Net
│   └── MultiSeg/                   # Lesion (LS) Segmentation code and models
│       ├── AttNetM.hdf5            # Trained weights for Lesion Attention Net
│       └── BUNETFOCALNOWEIGHTS.hdf5
├── Graphs/
│   ├── Graphs_final.ipynb          # Notebook for analyzing models and producing final plots
│   └── Graphs_final/               # Exported metric graphs (PDF/PNG format)
└── README.md                       # Project documentation
```

---

## Use Cases

1. **Automated DR Screening**: Quickly classifies patient fundus images into levels of Diabetic Retinopathy severity (e.g., No DR, Mild, Moderate, Severe, Proliferative DR).
2. **Macular Edema Risk Assessment**: Detects signs of DME to prevent severe macular damage.
3. **Clinical Assist Tool**: Generates segmentation overlays of blood vessels and micro-lesions, helping ophthalmologists visualize pathology areas more clearly.
4. **Research Benchmark**: Provides a comparative framework demonstrating the advantage of preprocessing via segmentation vs. raw image-based end-to-end learning.

---

## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.11.0
- Keras 2.11.0
- GPU acceleration (CUDA) is highly recommended for training.

### Installation

1. Clone or navigate to the project directory:
   ```bash
   cd "Code AI Model Diabetic Retinopathy"
   ```

2. Install core dependencies:
   ```bash
   pip install tensorflow==2.11.0 keras==2.11.0 opencv-python albumentations imgaug scikit-image scikit-learn pandas matplotlib openpyxl progressbar2 imageio
   ```
   *Note: If using `segmentation_models`, configure the framework backend:*
   ```python
   import os
   os.environ["SM_FRAMEWORK"] = "tf.keras"
   ```

---

## Pre-Trained Models & Weights

Due to size constraints, the trained model weights (`.hdf5` and `.h5` files) are not tracked in the Git repository. Instead, they are available in the **GitHub Releases** section of this repository. 

To use the models without retraining:
1. Navigate to the **Releases** page on GitHub.
2. Download the `.hdf5` weight files.
3. Place them in their respective local directories:
   * **MultiSeg Models** (Lesion segmentation): Place `AttNetM.hdf5` and `BUNETFOCALNOWEIGHTS.hdf5` into `Segmentation/MultiSeg/`
   * **RSEG Models** (Blood Vessel segmentation): Place `AttResUNET.hdf5`, `AttResUNET2.hdf5`, and `retina_AttentionRESUnet_150epochs.hdf5` into `Segmentation/RSEG/`
4. Downstream grading networks will automatically locate and load these pre-trained weights for evaluation.

---

## Workflow Guide

### 1. Data Preprocessing & Segmentation
Open and run either:
- [Unet_VesselSegmentation_preprocessing.ipynb](file:///Users/gvadrevu/Downloads/Code%20AI%20Model%20Diabetic%20Retinopathy/Segmentation/RSEG/Unet_VesselSegmentation_preprocessing.ipynb)
- [Unet_VesselSegmentation_preprocessing-new.ipynb](file:///Users/gvadrevu/Downloads/Code%20AI%20Model%20Diabetic%20Retinopathy/Segmentation/RSEG/Unet_VesselSegmentation_preprocessing-new.ipynb)

**Steps Included:**
- Loads raw fundus photographs and their ground truth masks.
- Applies image augmentations (flips, rotations, elastic transforms, CLAHE contrast adjustments).
- Trains the `Attention_ResUNet` or U-Net model.
- Saves segmented blood vessel results to output directories for the classification pipeline.

### 2. Diagnostic & Prognostic Grading
Refer to the notebooks inside `DRGrading` (found in the checkpoints folder or project source files) to train the classification networks.
- Load pre-segmented images (either `bvpath` or `lsp2path`).
- Train the transfer learning InceptionV3-Attention model.
- Run stacked ensembling classifiers (Logistic Regression, SVM) on predictions from individual classifiers.

### 3. Evaluation & Metrics Visualization
Open and execute [Graphs_final.ipynb](file:///Users/gvadrevu/Downloads/Code%20AI%20Model%20Diabetic%20Retinopathy/Graphs/Graphs_final.ipynb) to load training logs from `csv/` subfolders and generate final validation curves.

---

## Results & Performance

Below is an overview of the performance achieved by each network configuration as detailed in the graphs notebook:

* **Lesion-based Diagnostic Model**: **94.3%** classification accuracy.
* **Macular Edema (DME) Prognostic Model**: **92.4%** validation accuracy.
* **Blood Vessel-based Diagnostic Model**: **91.5%** classification accuracy.
* **Control Model (trained on raw images)**: **63.3%** accuracy.

> [!NOTE]
> The performance metrics validate that preprocessing images using custom segmentation networks (BV & LS) highlights key clinical markers, allowing downstream grading networks to achieve significantly higher classification accuracy compared to standard black-box training.
