# SEMICON India Hackathon 2026 - KLA Track

**AI-Based Restoration of Degraded Images for Semiconductor Inspection**

## Team: Phantom Protocol

| Name | Role |
|------|------|
| Tanmay Verma | Team Leader |
| Karan Agrawal | Member |
| Samridh Nautiyal | Member |
| Yuvraj Sharma | Member |

**College:** SRM University, Sonipat

## Problem Statement

Semiconductor wafers undergo inspection at multiple stages of manufacturing. These inspection images often suffer from:
- **Noise** from sensors and environmental factors
- **Low resolution** due to hardware limitations
- **Degradation** from storage and transmission

**Goal:** Build an AI model that restores degraded 128x128 noisy images to clean 256x256 high-resolution outputs.

## Our Solution: RestoreNet

A deep learning model combining:
- **8 Residual Blocks** for deep feature extraction
- **Channel Attention** for feature refinement
- **PixelShuffle (2x upsampling)** for super-resolution

### Architecture
```
Input (128x128) -> Conv -> 8x ResidualBlock -> ChannelAttention -> Conv -> PixelShuffle(2x) -> Conv -> Output (256x256)
```

### Key Specs
- **Parameters:** ~1.2M
- **Input:** 128x128 grayscale
- **Output:** 256x256 grayscale
- **Training:** 450 epochs on 3200 image pairs
- **Loss:** L1 Loss
- **Optimizer:** Adam (lr=2e-4)

## Results

| Metric | Value |
|--------|-------|
| Training Images | 3,200 pairs |
| Test Images | 400 images |
| Training Epochs | 450 |
| PSNR (Training) | ~74+ dB |

## Project Structure

```
semicon-kla-restoration/
├── models/
│   ├── model.py                 # RestoreNet architecture
│   └── checkpoint_epoch450.pth  # Trained model weights
├── scripts/
│   └── dataset.py               # KLA dataset loader
├── train.py                     # Training script
├── evaluation.py                # KLA evaluation script
├── restore_menu.py              # Interactive restoration tool
└── requirements.txt             # Dependencies
```

## How to Use

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Restore Images (Interactive)
```bash
python restore_menu.py
```
- Option 1: Single image restoration
- Option 2: Batch folder restoration

### 3. Run Evaluation
```bash
python evaluation.py --input_dir <test_noisy_lr> --output_dir <output>
```

### 4. Train (Optional)
```bash
python train.py
```

## Dataset

Download from: [Google Drive - KLA Dataset](https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6)

- `train.zip`: 3200 training pairs (GT: 256x256, NoisyLR: 128x128)
- `Test_NoisyLR.zip`: 400 test images (128x128)

## License

This project is developed for SEMICON India Hackathon 2026.
