# DiffPoseNet++

**DiffPoseNet++: Cheirality-Aware Contrastive Learning for Robust Pose Estimation**  
*Ebubekir Karamustafa, Emircan Kocaturk, Ilkin Umut Melanlioglu*  
COMP547 Deep Unsupervised Learning, Spring 2025

---

## Overview

DiffPoseNet++ is an enhanced deep learning-based visual odometry system, extending the original DiffPoseNet architecture. It combines robust pose estimation, normal flow prediction, and a differentiable cheirality layer to improve generalization and stability in camera pose estimation tasks. Our contributions include significant architectural upgrades to the PoseNet and NFlowNet modules, along with the integration of contrastive learning and attention mechanisms.

## Architecture

The system is built from three core modules:

### 1. PoseNet / PoseNet+ / PoseNet++

- **PoseNet**: The baseline pose regressor, originally based on a VGG-16 CNN and stacked LSTM layers, predicts relative camera motion between frames.
- **PoseNet+**: Improves upon PoseNet by replacing VGG-16 with a DINOv2 visual encoder, deepening and making LSTM bidirectional, and adding multi-head attention and dedicated MLPs for translation and rotation (quaternion) outputs.
- **PoseNet++**: Further enhances PoseNet+ with contrastive learning, encouraging better alignment and discrimination of temporal features, using both spatial-temporal and positive/negative sequence contrastive pairs.

All versions output translation vectors and quaternion rotations, with custom loss balancing and optional uncertainty weighting.

### 2. NFlowNet

- A lightweight U-Net inspired network that predicts the normal flow (optical flow projected onto the image gradient) between consecutive frames.
- Designed to provide strong motion cues for self-supervised pose refinement.
- Features residual connections within encoder/decoder blocks, without skip connections or attention modules (for efficiency).

### 3. Cheirality Layer

- Implements a differentiable geometric constraint: the depth positivity (cheirality) condition, ensuring all points are in front of the camera.
- Acts as an optimization block that refines PoseNet's predictions using NFlowNet’s normal flow outputs.
- Utilizes implicit differentiation and a quasi-Newton optimizer (L-BFGS) for end-to-end training.

## Repository Structure

```
dataset/          # Tools for downloading and loading the TartanAir dataset
  └── ...         # Dataset loader and helpers

posnet/           # PoseNet models and training scripts
  ├── model.py            # Original PoseNet & PoseNet+ model
  ├── model-improved.py   # PoseNet and PoseNet++ architectures
  ├── train.py            # Training script for PoseNet
  ├── train-per-sequence.py # (Experimental, not required for standard training)
  └── ...                 # Utilities

nflownet/         # NFlowNet models and training scripts
  ├── model.py            # Original NFlowNet model
  ├── train.py            # Training script for NFlowNet
  └── ...                 # Utilities

cheirality/        # Implementation of the cheirality (depth positivity) layer
  └── ...         # Optimization and constraint code

other/            # Additional utilities, scripts, and documentation
```

## Dataset

- **TartanAir Dataset**: The project uses the TartanAir dataset for both training and evaluation. Our dataset loader in the `dataset/` folder can be used to automate downloading and preprocessing.
- Only left camera images, their positions, optical flows, and masks are used for training and evaluation.

## Key Improvements

- **Visual Encoder**: Upgraded from VGG-16 to DINOv2 for improved spatial feature extraction.
- **Temporal Modeling**: Deeper, bidirectional LSTM with added multi-head attention.
- **Loss Balancing**: Experimented with uncertainty weighting, manual tuning for better translation/quaternion learning.
- **Contrastive Learning**: Explored two approaches to align temporal and spatial features and to distinguish between similar/dissimilar frame sequences.
- **Normal Flow Masking**: Improved preprocessing for sharper, less noisy normal flow maps.
- **Future Directions**: Plans to add skip connections and attention to NFlowNet, and to further refine the cheirality layer.

## Experimental Results

- Improved translation and overall loss in PoseNet+ compared to the original.
- PoseNet++ with contrastive learning provided insights but did not surpass PoseNet+ in final loss.
- NFlowNet achieved strong visual and quantitative results on normal flow prediction.
- Cheirality layer implementation faced optimization challenges, which are under further investigation.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Contact

For questions or collaboration, contact:  
- Ebubekir Karamustafa: ekaramustafa20@ku.edu.tr  
---

For more details, refer to our project report
