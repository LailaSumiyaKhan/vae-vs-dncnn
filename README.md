# VAE vs DnCNN: A Comparative Study for Image Denoising

A reproducible comparison of the Variational Autoencoder (VAE) and the
Denoising Convolutional Neural Network (DnCNN) for grayscale image denoising,
evaluated on the BSD68 benchmark.

## Results Summary

| Noise | σ | Model | PSNR (dB) | SSIM | LPIPS |
|-------|---|-------|-----------|------|-------|
| Gaussian | 15 | DnCNN | **31.58** | **0.891** | **0.105** |
| Gaussian | 15 | VAE   | 23.14 | 0.554 | 0.577 |
| Gaussian | 25 | DnCNN | **28.96** | **0.822** | **0.169** |
| Gaussian | 25 | VAE   | 22.17 | 0.504 | 0.635 |
| Poisson  | 15 | DnCNN | **33.42** | **0.925** | **0.059** |
| Poisson  | 15 | VAE   | 22.40 | 0.517 | 0.618 |
| Poisson  | 25 | DnCNN | **30.57** | **0.870** | **0.115** |
| Poisson  | 25 | VAE   | 22.18 | 0.507 | 0.624 |

DnCNN is **27× faster** at inference (1.50 ms vs 40.61 ms per image) and uses
**10.7× fewer parameters** (556K vs 5.97M).

## Project Structure

```
vae_vs_dcnn/
├── main.py                  # Entry point (train / eval / visualize)
├── src/
│   ├── config.py            # Hyperparameters and paths
│   ├── dataset.py           # BSD300/BSD68 loading and patch extraction
│   ├── noise.py             # Gaussian, Poisson, speckle synthesis
│   ├── models/
│   │   ├── dncnn.py         # 17-layer residual CNN
│   │   └── vae.py           # Convolutional VAE with 256-d latent space
│   ├── trainer.py           # Training loops for both models
│   ├── evaluator.py         # PSNR / SSIM / LPIPS evaluation on BSD68
│   ├── metrics.py           # Metric utilities
│   └── visualizer.py       # Figure generation
├── visualizations/          # Scripts for each paper figure
├── report/
│   ├── paper.tex            # IEEE-format paper (LaTeX)
│   ├── references.bib       # Bibliography
│   └── figures/             # Generated PDF figures
├── data/
│   ├── train/               # BSD300 training images
│   └── test/                # BSD68 test images
└── results/
    ├── checkpoints/         # Saved model weights
    ├── metrics/             # JSON/CSV evaluation results
    └── figures/             # Output figures
```

## Setup

**Requirements:** Python ≥ 3.12, CUDA-capable GPU recommended.

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install torch torchvision numpy Pillow matplotlib scikit-image lpips tqdm pandas seaborn scipy
```

### Data

Download the BSD500 dataset and split into training and test sets:

```
data/train/   ← 300 images from BSD300
data/test/    ← 68 images from BSD68
```

## Usage

```bash
# Train all models (3 noise types × 3 sigma levels × 2 architectures)
python main.py train

# Evaluate all trained models on BSD68
python main.py eval

# Generate all paper figures
python main.py visualize

# Full pipeline: train → eval → visualize
python main.py all
```

### Optional flags

```bash
--noise-type gaussian     # restrict to one noise type (gaussian | poisson | speckle)
--sigma 25                # restrict to one sigma level (15 | 25 | 50)
--epochs 50               # override epoch count (default: 50)
--batch-size 128
--workers 4
```

### Examples

```bash
# Train only for Gaussian noise at σ=25, 10 epochs
python main.py train --noise-type gaussian --sigma 25 --epochs 10

# Evaluate only for Poisson noise
python main.py eval --noise-type poisson
```

## Models

### DnCNN
- 17-layer fully convolutional network
- Residual learning: predicts noise, subtracts from input
- Conv + BN + ReLU blocks; 64 filters of size 3×3
- Processes images of any resolution end-to-end

### VAE
- Encoder: 4 strided Conv-BN-LeakyReLU blocks → 256-d latent space
- Decoder: symmetric transposed convolution blocks
- Applied patch-by-patch (64×64 tiles) at inference
- KL weight β = 1e-4

## Training Details

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 with cosine annealing |
| Batch size | 128 |
| Patch size | 64×64 |
| Patches per image | 200 |
| GPU | NVIDIA RTX 3060 (12 GB) |
| Framework | PyTorch 2.12 + CUDA 12.6 |
