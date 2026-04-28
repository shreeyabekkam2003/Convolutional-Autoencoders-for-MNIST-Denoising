# Convolutional Autoencoders for MNIST Denoising

> Comparing two Convolutional Autoencoder architectures to understand how model complexity affects image reconstruction quality and noise removal on the MNIST dataset.

## Overview

This notebook implements and compares two **Convolutional Autoencoder (CAE)** configurations for the task of **image denoising** on MNIST handwritten digits. The central question explored is:

> *Does a more complex model always perform better, or can a lighter architecture generalize more robustly when noise is introduced?*

Both models are trained on clean MNIST images and then evaluated on artificially **Gaussian-noisy** versions of the test set to assess their denoising capability.

---

## Key Finding

> ⚡ **A simpler model denoises better.**

| Model | Train Loss | Denoising Quality |
|---|---|---|
| `AutoEncoder_config1` (more filters) | ~0.002 | ❌ Struggles with noise |
| `AutoEncoder_config2` (fewer filters) | ~0.010 | ✅ Superior denoising |

Despite achieving lower reconstruction loss on clean images, **Config 1 overfits to fine-grained pixel details**, making it sensitive to noise. **Config 2**, with fewer parameters, is forced to learn more abstract and robust features — resulting in better noise suppression.

---

## Dataset

| Property | Value |
|---|---|
| Dataset | MNIST (handwritten digits) |
| Train / Validation / Test | Standard PyTorch split |
| Image Shape | 1 × 28 × 28 (grayscale) |
| Normalization | [0, 1] |
| Batch Size | 32 |
| Noise Type | Gaussian (noise factor = 0.4), clipped to [0, 1] |

---

## Model Architectures

Both models share the same overall encoder-decoder structure using `Conv2d` / `ConvTranspose2d` layers with **LeakyReLU** activations and a **Sigmoid** output.

### Config 1 — Higher Capacity

```
Encoder:
  Conv2d(1 → 8,  kernel=3, stride=2, pad=1)  → [8, 14, 14]  + LeakyReLU(0.01)
  Conv2d(8 → 4,  kernel=3, stride=2, pad=1)  → [4,  7,  7]  + LeakyReLU(0.01)

Decoder:
  ConvTranspose2d(4 → 8,  kernel=3, stride=2, pad=1)  → [8, 14, 14]  + LeakyReLU(0.01)
  ConvTranspose2d(8 → 1,  kernel=3, stride=2, pad=1)  → [1, 28, 28]  + Sigmoid
```

### Config 2 — Lower Capacity (Better Denoiser)

```
Encoder:
  Conv2d(1 → 4,  kernel=3, stride=2, pad=1)  → [4, 14, 14]  + LeakyReLU(0.01)
  Conv2d(4 → 2,  kernel=3, stride=2, pad=1)  → [2,  7,  7]  + LeakyReLU(0.01)

Decoder:
  ConvTranspose2d(2 → 4,  kernel=3, stride=2, pad=1)  → [4, 14, 14]  + LeakyReLU(0.01)
  ConvTranspose2d(4 → 1,  kernel=3, stride=2, pad=1)  → [1, 28, 28]  + Sigmoid
```

> Both architectures use custom `Reshape` and `Trim` utility modules to handle tensor reshaping in the decoder.

---

## Training

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Epochs | 10 |
| Batch Size | 32 |
| Loss Function | MSE (Mean Squared Error) |
| Random Seed | 123 |
| Device | CUDA if available, else CPU |

Training is performed on **clean MNIST images** for both configs. No noise is added during training — the denoising capability emerges from the model's learned compressed representation.

---

## Evaluation & Denoising

### Clean Reconstruction

Both models are evaluated on their ability to reconstruct clean training images. Training loss curves are plotted and sample reconstructions are visualized.

### Denoising Test

Gaussian noise is added to test images using:

```python
def add_noise(images, noise_factor=0.4):
    noise = noise_factor * torch.randn(*images.shape)
    noisy_imgs = images + noise
    return torch.clamp(noisy_imgs, 0., 1.)
```

Both models then reconstruct the noisy images. A side-by-side visualization displays 10 samples across 4 columns:

| Original | Noisy | Config 1 Output | Config 2 Output |
|---|---|---|---|

---

## Results Summary

| Metric | Config 1 | Config 2 |
|---|---|---|
| Parameters | More | Fewer |
| Training Loss | ~0.002 (lower) | ~0.010 (higher) |
| Clean Reconstruction | Excellent | Good |
| Noise Removal | Poor | **Excellent** |
| Generalization | Overfits noise | **Robust** |

**Interpretation:** Config 2's bottleneck (2 channels at 7×7 = **98 values**) forces extreme compression, discarding noise as irrelevant variation. Config 1's larger bottleneck (4 channels at 7×7 = **196 values**) memorizes noise patterns during training.

---

## Requirements

```bash
pip install torch torchvision matplotlib numpy
```

The notebook also depends on these helper modules (assumed to be in the working directory):

| Module | Purpose |
|---|---|
| `helper_data.py` | `get_dataloaders_mnist()` — MNIST DataLoader factory |
| `helper_train.py` | `train_autoencoder_v1()` — training loop |
| `helper_utils.py` | `set_deterministic()`, `set_all_seeds()` |
| `helper_plotting.py` | `plot_training_loss()`, `plot_generated_images()` |

---

## Usage

1. Ensure all `helper_*.py` files are in your working directory.
2. Run cells sequentially. Config 1 trains first, then Config 2.
3. The denoising comparison visualization runs automatically at the end.

```python
# Quick model instantiation
model = AutoEncoder_config2()
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Train
log_dict = train_autoencoder_v1(num_epochs=10, model=model,
                                 optimizer=optimizer, device=DEVICE,
                                 train_loader=train_loader)
```

---

## Project Structure

```
Convolutional_Autoencoders_for_MNIST_Denoising.ipynb
│
├── Settings & Seed Control
├── Dataset Loading (MNIST via helper_data)
├── AutoEncoder_config1
│   ├── Model Definition
│   ├── Training (10 epochs)
│   └── Clean Reconstruction Visualization
├── AutoEncoder_config2
│   ├── Model Definition
│   ├── Training (10 epochs)
│   └── Clean Reconstruction Visualization
└── Denoising Comparison
    ├── Gaussian Noise Generation
    ├── Side-by-side Reconstruction: Config 1 vs Config 2
    └── Analysis & Conclusions
```

---

## Concepts Demonstrated

| Concept | Description |
|---|---|
| Convolutional Autoencoder | Encoder-decoder with conv layers for spatial feature learning |
| Bottleneck Representation | Compressed latent space forces learning of essential features |
| Denoising Autoencoder | Using a trained CAE to remove noise at inference |
| Capacity vs Generalization | Trade-off between model size and robustness |
| LeakyReLU | Prevents dying ReLU problem in conv layers |
| Gaussian Noise Injection | Synthetic noise added at inference for robustness testing |
| Deterministic Training | Fixed seeds for reproducibility across runs |
