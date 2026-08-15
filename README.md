# SEMICON India Hackathon 2026 - KLA Track

**AI-Based Restoration of Degraded Images for Semiconductor Inspection**

---

## Team: Phantom Protocol

| Name | Role | GitHub |
|------|------|--------|
| Tanmay Verma | Team Leader | [@lords-coder](https://github.com/lords-coder) |
| Karan Agrawal | Lead developer and presenter | [@Karanagrawa1955](https://github.com/Karanagrawa1955) |
| Yuvraj Sharma | Full-stack engineer | [@yuvrajsharma01official](https://github.com/yuvrajsharma01official) |
| Samridh Nautiyal | Documentation lead | [@QuantumSyntax27](https://github.com/QuantumSyntax27) |

**Mentor--** Dr. Brajesh

**College:** SRM University, Sonipat, Haryana

**Hackathon Portal:** [i4C SEMICON 2026](https://hackathon2026.i4c.in/)

**Problem Statement:** [SEMICON India 2026 - KLA Track](https://www.semi.org/en/india-semicon/semicon-india-2026)

---

## Problem Statement

Semiconductor wafers undergo visual inspection at multiple stages of manufacturing. These inspection images often suffer from:

- **Noise** introduced by imaging sensors, electromagnetic interference, and environmental factors
- **Low resolution** due to hardware limitations and cost constraints on inspection equipment
- **Degradation** from data compression, storage, and transmission across systems

Manual inspection of degraded images is time-consuming, error-prone, and does not scale with modern semiconductor production volumes.

**Goal:** Build an AI model that takes degraded 128x128 noisy grayscale images and restores them to clean 256x256 high-resolution outputs, improving both visual quality and defect detection accuracy.

---

## Our Solution: RestoreNet

We designed **RestoreNet**, a lightweight convolutional neural network for single-image super-resolution and denoising, specifically tailored for semiconductor inspection images.

### Architecture Overview

```
Input (1x128x128)
    │
    ▼
Conv2d(1→64, 3x3)
    │
    ▼
8x ResidualBlock (Conv→LeakyReLU→Conv + Skip Connection)
    │
    ▼
Channel Attention (AdaptiveAvgPool→FC→Sigmoid→Scale)
    │
    ▼
Conv2d(64→64, 3x3) + Skip Connection from first conv
    │
    ▼
PixelShuffle Upsample (2x): Conv2d(64→256, 3x3) → Reshape → LeakyReLU
    │
    ▼
Conv2d(64→1, 3x3)
    │
    ▼
Output (1x256x256)
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **Residual Blocks (x8)** | Deep feature extraction with skip connections to preserve information |
| **Channel Attention** | Dynamically weights feature channels by importance for better feature refinement |
| **PixelShuffle (2x)** | Efficient sub-pixel convolution for learnable upsampling from 128→256 |
| **Skip Connection** | Direct connection from first conv to upsample input, preserving low-level features |

### Model Specifications

| Spec | Value |
|------|-------|
| Architecture | RestoreNet (Custom CNN) |
| Parameters | ~1.2M |
| Input Size | 128x128 grayscale (1 channel) |
| Output Size | 256x256 grayscale (1 channel) |
| Upsampling Method | PixelShuffle (2x sub-pixel convolution) |
| Activation | LeakyReLU (negative slope=0.2) |
| Attention | Channel Attention with reduction ratio=16 |

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch 2.12+ (CUDA 12.8) |
| GPU | NVIDIA GeForce RTX 5060 (Laptop) |
| Epochs | 450 |
| Batch Size | 16 |
| Loss Function | L1 Loss (Mean Absolute Error) |
| Optimizer | Adam |
| Learning Rate | 2e-4 |
| Data Augmentation | Random horizontal flip, vertical flip, 90-degree rotation |
| Checkpointing | Every 50 epochs |

### Training Progress

| Epoch | Loss |
|-------|------|
| 50 | — |
| 100 | — |
| 200 | — |
| 300 | — |
| 450 | Best |

---

## Results

| Metric | Value |
|--------|-------|
| Training Image Pairs | 3,200 |
| Test Images | 400 |
| Training Epochs | 450 |
| PSNR (Training) | ~74+ dB |
| Inference Speed | <50ms per image (GPU) |

---

## Project Structure

```
semicon-restoration/
├── models/
│   ├── model.py                  # RestoreNet architecture definition
│   ├── checkpoint_epoch450.pth   # Best trained model weights (epoch 450)
│   └── final.pth                 # Final model snapshot
├── scripts/
│   └── dataset.py                # KLA dataset loader (.npy format)
├── train.py                      # Training script (resumable)
├── evaluation.py                 # KLA evaluation script (batch inference)
├── restore_menu.py               # Interactive restoration tool (single/batch)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/lords-coder/semicon-restoration.git
cd semicon-restoration
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** For GPU acceleration, install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. Download the Dataset

Download from: [Google Drive - KLA Dataset](https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6)

Extract to the project root:
```
semicon-restoration/
├── train_raw/
│   └── train/
│       ├── GT/           # 3200 ground truth images (256x256 .npy)
│       └── NoisyLR/      # 3200 degraded images (128x128 .npy)
└── test_noisy_lr/
    └── NoisyLR/          # 400 test images (128x128 .npy)
```

### 4. Restore Images (Interactive)

```bash
python restore_menu.py
```

**Menu Options:**
- **Option 1:** Single image restoration (supports .npy, .png, .jpg)
- **Option 2:** Batch folder restoration (process all images in a folder)
- **Option 3:** Exit

**Output:**
- Restored image saved as `.png`
- Side-by-side comparison image (degraded vs restored)

### 5. Run Batch Evaluation

```bash
python evaluation.py --input_dir test_noisy_lr/NoisyLR --output_dir output/
```

This processes all `.npy` files in the input directory and saves restored `.png` images to the output directory.

### 6. Train from Scratch (Optional)

```bash
# Train from epoch 0
python train.py

# Resume from checkpoint
python train.py --resume models/checkpoint_epoch450.pth
```

Checkpoints are saved every 50 epochs in `models/`.

---

## Dataset Format

| Dataset | Format | Size | Count | Description |
|---------|--------|------|-------|-------------|
| Ground Truth (GT) | `.npy` (numpy) | 256x256 uint8 | 3,200 | Clean reference images |
| NoisyLR | `.npy` (numpy) | 128x128 uint8 | 3,200 | Degraded low-resolution inputs |
| Test NoisyLR | `.npy` (numpy) | 128x128 uint8 | 400 | Test set for evaluation |

---

## Technology Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.11** | Programming language |
| **PyTorch 2.12+** | Deep learning framework |
| **CUDA 12.8** | GPU acceleration |
| **NumPy** | Numerical computation, .npy file I/O |
| **OpenCV** | Image processing, CLAHE enhancement |
| **tqdm** | Training progress visualization |

---

## How It Works

1. **Input:** A 128x128 degraded grayscale semiconductor inspection image
2. **Feature Extraction:** The model extracts deep features through 8 residual blocks
3. **Channel Attention:** Important feature channels are weighted higher for better reconstruction
4. **Upsampling:** PixelShuffle learns to reconstruct a 256x256 image from the 128x128 features
5. **Output:** A clean, high-resolution 256x256 restored image

---

## License

This project was developed for the **SEMICON India Hackathon 2026 - KLA Track**.

**Problem:** AI-Based Restoration of Degraded Images for Semiconductor Inspection

**Portal:** [hackathon2026.i4c.in](https://hackathon2026.i4c.in/)

---

## Acknowledgments

- [KLA Corporation](https://www.kla.com/) for the problem statement and dataset
- [SEMICON India 2026](https://www.semi.org/en/india-semicon/semicon-india-2026) for organizing the hackathon
- [i4C](https://hackathon2026.i4c.in/) for the registration portal
- [PyTorch](https://pytorch.org/) for the deep learning framework
